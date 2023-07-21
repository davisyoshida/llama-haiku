"""Adapted from torch implementation at https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""
from collections import namedtuple
from functools import partial
import logging
import warnings

from flash_attention_jax import causal_flash_attention
import haiku as hk
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import jmp

logger = logging.getLogger(__name__)

LlamaConfig = namedtuple('LlamaConfig', [
    'vocab_size',
    'hidden_size',
    'intermediate_size',
    'max_position_embeddings',
    'num_attention_heads',
    'rms_norm_eps',
    'num_hidden_layers',
])

def get_dtype():
    policy = hk.mixed_precision.current_policy()
    return policy.compute_dtype if policy else jnp.float32

class ConfigModule(hk.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

class VarianceOnlyLayerNorm(ConfigModule):
    """LayerNorm but without subtracting the mean"""
    def __call__(self, x):
        variance = jnp.mean((x.astype(jnp.float32) ** 2), axis=-1, keepdims=True)
        scale = hk.get_parameter(
            'weight',
            x.shape[-1:],
            init=hk.initializers.Constant(1.)
        )
        x = x * jax.lax.rsqrt(variance + self.config.rms_norm_eps)
        return (x * scale).astype(get_dtype())

def rotate_half(x):
    n = x.shape[-1] // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    return jnp.concatenate([-x2, x1], axis=-1)

class LlamaRotaryEmbedding(ConfigModule):
    def __init__(self, config, base=10000):
        super().__init__(config=config)
        self.base = base
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        dtype = get_dtype()

        inv_freq = 1 / (self.base ** (jnp.arange(0, head_dim, 2.0, dtype=jnp.float32) / head_dim))
        freqs = jnp.arange(self.config.max_position_embeddings)[:, None] * inv_freq[None, :]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin = jnp.sin(emb).astype(dtype)
        self.cos = jnp.cos(emb).astype(dtype)

    def __call__(self, queries, keys, pos_ids):
        cos = self.cos[pos_ids]
        sin = self.sin[pos_ids]
        q_emb = queries * cos + rotate_half(queries) * sin
        k_emb = keys * cos + rotate_half(keys) * sin
        return q_emb, k_emb

class LlamaMLP(ConfigModule):
    def __call__(self, x):
        gate = jax.nn.silu(hk.Linear(self.config.intermediate_size, name='gate_proj', with_bias=False)(x))
        val = hk.Linear(self.config.intermediate_size, name='up_proj', with_bias=False)(x)

        return hk.Linear(self.config.hidden_size, name='down_proj', with_bias=False)(gate * val)

_weighted_sum_values = partial(jnp.einsum, 'hts,hsd->htd')
class LlamaAttention(ConfigModule):
    def __init__(self, pos_emb, use_flash_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb
        self.use_flash_attention = use_flash_attention
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past,
        no_cache_update=False
    ):

        # n_head x time x head_dim
        queries, keys, values = [
            checkpoint_name(
                hk.Linear(self.config.hidden_size, name=f'{name}_proj', with_bias=False)(hidden_states).reshape(
                    -1, self.config.num_attention_heads, self.head_dim
                ).transpose(1, 0, 2),
                name=f'llama_{name}_proj'
            )
            for name in 'qkv'
        ]

        queries, keys = self.pos_emb(queries, keys, position_ids)

        q_len = hidden_states.shape[-2]

        if no_cache_update:
            output, new_past = self._attention_with_no_cache_update(queries, keys, values, past, attention_mask, q_len)
        else:
            output, new_past = self._attention_with_cache(queries, keys, values, past, attention_mask, q_len)

        assert output.shape == (self.config.num_attention_heads, q_len, self.head_dim)
        output = output.transpose(1, 0, 2).reshape(-1, self.config.hidden_size)

        result = hk.Linear(self.config.hidden_size, name='o_proj', with_bias=False)(output.reshape(-1, self.config.hidden_size))
        return result, new_past

    def _attention_with_cache(self, queries, keys, values, past, attention_mask, q_len):
        past_kv, past_length = past
        if past_kv is not None:
            keys, values = [
                jax.lax.dynamic_update_slice_in_dim(p_tensor, curr_tensor, past_length, axis=1)
                for p_tensor, curr_tensor in zip(past_kv, [keys, values])
            ]

        # should output batch x heads x seq x dim
        if self.use_flash_attention:
            output = causal_flash_attention(queries, keys, values)
        else:
            attention_weights = self._query_key_dot(queries, keys)
            attention_weights += attention_mask
            out_type = attention_weights.dtype
            attention_weights = jax.nn.softmax(attention_weights.astype(jnp.float32), axis=-1)
            attention_weights = attention_weights.astype(out_type)

            expected_kv_size = q_len if past_kv is None else past_kv[0].shape[1]
            assert attention_weights.shape == (self.config.num_attention_heads, q_len, expected_kv_size), f'{attention_weights.shape} != {(self.config.num_attention_heads, q_len, expected_kv_size)}'

            output = _weighted_sum_values(attention_weights, values)
        return output, (keys, values)


    def _attention_with_no_cache_update(self, queries, keys, values, past, attention_mask, q_len):
        if self.use_flash_attention:
            warnings.warn('Cannot use flash attention when `updatekv_after_dot` is passed, using regular attention')

        past_kv, past_length = past

        past_dots = pv = None
        if past_kv is not None:
            pk, pv = past_kv
            past_dots = self._query_key_dot(queries, pk).astype(jnp.float32)

        out_dtype = queries.dtype

        present_dots = self._query_key_dot(queries, keys)
        current_weights = (present_dots + attention_mask).astype(jnp.float32)

        past_value_mean = None
        if past_dots is not None:
            # past dots is heads x queries x keys
            mask = jnp.where(jnp.arange(past_dots.shape[-1]) < past_length, 0, -jnp.inf)
            assert mask.shape == (past_dots.shape[-1],)
            past_dots += mask

            past_lse = jax.scipy.special.logsumexp(past_dots, axis=-1, keepdims=True)
            current_lse = jax.scipy.special.logsumexp(current_weights, axis=-1, keepdims=True)
            total_lse = jnp.logaddexp(past_lse, current_lse)

            past_weights = jnp.exp(past_dots - total_lse).astype(out_dtype)
            current_weights = jnp.exp(current_weights - total_lse).astype(out_dtype)

            past_value_mean = _weighted_sum_values(past_weights, pv)
            curr_value_mean = _weighted_sum_values(current_weights, values)

            value_mean = jnp.where(past_length == 0, curr_value_mean, curr_value_mean + past_value_mean)
        else:
            current_weights = jax.nn.softmax(current_weights, axis=-1).astype(out_dtype)
            value_mean = _weighted_sum_values(current_weights, values)

        assert value_mean.shape == (self.config.num_attention_heads, q_len, self.head_dim)
        return value_mean, (keys, values)

    def _query_key_dot(self, queries, keys):
        return jnp.einsum('htd,hsd->hts', queries, keys) / jnp.sqrt(self.head_dim)

class LLamaDecoderLayer(ConfigModule):
    def __init__(self, pos_emb, checkpoint_mlp=False, mlp_block_size=None, use_flash_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb
        self.checkpoint_mlp = checkpoint_mlp
        self.mlp_block_size = mlp_block_size
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past,
        **kwargs
    ):
        residual = hidden_states

        norm_output = VarianceOnlyLayerNorm(self.config, name='input_layernorm')(hidden_states)

        attention_layer = LlamaAttention(
            pos_emb=self.pos_emb,
            config=self.config,
            name='self_attn',
            use_flash_attention=self.use_flash_attention
        )
        if self.checkpoint_mlp:
            attention_layer = hk.remat(attention_layer)

        attn_output, present = attention_layer(
            norm_output,
            attention_mask,
            position_ids,
            past,
            **kwargs
        )
        residual += attn_output

        post_attn_norm_output = VarianceOnlyLayerNorm(self.config, name='post_attention_layernorm')(residual)
        mlp_layer = LlamaMLP(self.config)
        if self.checkpoint_mlp:
            mlp_layer = hk.remat(mlp_layer)

        if self.mlp_block_size is None:
            mlp_output = mlp_layer(post_attn_norm_output)
        elif post_attn_norm_output.shape[0] % self.mlp_block_size == 0:
            n_blocks = post_attn_norm_output.shape[0] // self.mlp_block_size
            inps = jnp.split(post_attn_norm_output, n_blocks, axis=0)
            outs = []
            for inp in inps:
                outs.append(mlp_layer(inp))
            mlp_output = jnp.concatenate(outs, axis=0)
        else:
            raise ValueError(f'MLP block size must be a divisor of input length {inp.shape[0]}')

        mlp_output = mlp_layer(post_attn_norm_output)
        residual += mlp_output

        return residual, present

def _make_causal_mask(input_indices, past_cache_size=None):
    seq_len = input_indices.shape[-1]
    kv_length = seq_len if past_cache_size is None else past_cache_size
    mask = jnp.full((seq_len, kv_length), -jnp.inf)
    mask = jnp.where(
        input_indices[:, None] >= jnp.arange(kv_length)[None, :],
        0.,
        mask
    )
    return mask

class LlamaModel(ConfigModule):
    def __init__(self, config, name='model'):
        super().__init__(config=config)

    def __call__(
        self,
        input_ids,
        past=None,
        past_cache_size=None,
        return_past=False,
        return_hidden=False,
        checkpoint=False,
        checkpoint_mlp=False,
        mlp_block_size=1,
        use_flash_attention=False,
        no_cache_update=False,
    ):
        inp_length = input_ids.shape[-1]
        if past is None:
            past_length = 0
            if past_cache_size is not None:
                cache_shape = (
                    2,
                    self.config.num_attention_heads,
                    past_cache_size,
                    self.config.hidden_size // self.config.num_attention_heads
                )
                past = [
                    jnp.zeros(
                        cache_shape,
                        dtype=get_dtype()
                     )
                    for _ in range(self.config.num_hidden_layers)
                ]
            else:
                past = [None] * self.config.num_hidden_layers
                past_cache_size = 0
            indices = jnp.arange(inp_length)
        else:
            past, past_length = past
            if past_cache_size != past[0][0].shape[1]:
                logger.warning(f'past_cache_size {past_cache_size} != {past[0][0].shape[1]}, passed value will be ignored')
                past_cache_size = past[0][0].shape[1]

            indices = jax.lax.dynamic_slice_in_dim(jnp.arange(past_cache_size), past_length, inp_length)

        if no_cache_update:
            attention_mask = _make_causal_mask(
                jnp.arange(inp_length),
                past_cache_size=None,
            )
        else:
            attention_mask = _make_causal_mask(
                indices,
                past_cache_size=past_cache_size if past_cache_size else None,
            )

        full_seq_length = inp_length + past_length
        wte = hk.get_parameter(
            'embed_tokens_weight',
            shape=(self.config.vocab_size, self.config.hidden_size),
            init=hk.initializers.RandomUniform(-0.02, 0.02)
        )
        hidden_states = wte[input_ids,]

        pos_emb = LlamaRotaryEmbedding(config=self.config)

        presents = []
        hidden = []
        for layer_num, layer_past in enumerate(past):
            if return_hidden:
                hidden.append(hidden_states)
            layer = LLamaDecoderLayer(
                config=self.config,
                pos_emb=pos_emb,
                name=f'layer_{layer_num}',
                checkpoint_mlp=checkpoint_mlp,
                mlp_block_size=mlp_block_size,
                use_flash_attention=use_flash_attention
            )
            if checkpoint:
                layer = hk.remat(
                    layer,
                    static_argnums=(3,),
                )
            hidden_states, present = layer(
                hidden_states,
                attention_mask,
                indices,
                (layer_past, past_length),
                no_cache_update=no_cache_update
            )
            hidden_states = jax.ad_checkpoint.checkpoint_name(hidden_states, f'llama_hidden_state_{layer_num}')
            if return_past:
                presents.append(present)

        norm_out = VarianceOnlyLayerNorm(self.config, name='norm')(hidden_states)

        ret = {}

        if return_hidden:
            hidden.append(norm_out)
            ret['hidden'] = hidden


        logits = hk.Linear(self.config.vocab_size, name='lm_head', with_bias=False)(norm_out)
        ret['logits'] = logits

        if return_past:
            ret['past'] = (presents, full_seq_length)

        return ret
