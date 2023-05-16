"""Adapted from torch implementation at https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""
from collections import namedtuple
import logging

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
        # TODO: Decide if use_fast_variance technique from hk.LayerNorm is helpful
        variance = jnp.var(x.astype(jnp.float32), axis=-1, keepdims=True)
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

        inv_freq = 1 / (self.base ** (jnp.arange(0, head_dim, 2.0, dtype=dtype) / head_dim))
        freqs = jnp.arange(self.config.max_position_embeddings)[:, None] * inv_freq[None, :]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin = jnp.sin(emb)
        self.cos = jnp.cos(emb)

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

class LlamaAttention(ConfigModule):
    def __init__(self, pos_emb, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past
    ):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # n_head x time x head_dim
        queries, keys, values = [
            checkpoint_name(
                hk.Linear(self.config.hidden_size, name=f'{name}_proj', with_bias=False)(hidden_states).reshape(
                    -1, self.config.num_attention_heads, head_dim
                ).transpose(1, 0, 2),
                name=f'llama_{name}_proj'
            )
            for name in 'qkv'
        ]

        q_len = hidden_states.shape[-2]

        past_kv, past_length = past

        queries, keys = self.pos_emb(queries, keys, position_ids)

        if past_kv is not None:
            keys, values = [
                jax.lax.dynamic_update_slice_in_dim(p_tensor, curr_tensor, past_length, axis=1)
                for p_tensor, curr_tensor in zip(past_kv, [keys, values])
            ]

        attention_weights = jnp.einsum('htd,hsd->hts', queries, keys) / jnp.sqrt(head_dim)
        attention_weights += attention_mask
        attention_weights = jax.nn.softmax(attention_weights, axis=-1)

        expected_kv_size = hidden_states.shape[-2] if past_kv is None else past_kv[0].shape[1]
        assert attention_weights.shape == (self.config.num_attention_heads, q_len, expected_kv_size), f'{attention_weights.shape} != {(self.config.num_attention_heads, q_len, expected_kv_size)}'

        output = jnp.einsum('hts,hsd->htd', attention_weights, values)
        assert output.shape == (self.config.num_attention_heads, q_len, head_dim)
        output = output.transpose(1, 0, 2).reshape(-1, self.config.hidden_size)

        result = hk.Linear(self.config.hidden_size, name='o_proj', with_bias=False)(output.reshape(-1, self.config.hidden_size))
        new_past = (keys, values)
        return result, new_past

class LLamaDecoderLayer(ConfigModule):
    def __init__(self, pos_emb, checkpoint_mlp=False, mlp_block_size=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb
        self.checkpoint_mlp = checkpoint_mlp
        self.mlp_block_size = mlp_block_size

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past,
    ):
        residual = hidden_states

        norm_output = VarianceOnlyLayerNorm(self.config, name='input_layernorm')(hidden_states)

        attention_layer = LlamaAttention(pos_emb=self.pos_emb, config=self.config, name='self_attn')
        if self.checkpoint_mlp:
            attention_layer = hk.remat(attention_layer)

        attn_output, present = attention_layer(
            norm_output,
            attention_mask,
            position_ids,
            past
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
                logger.warning(f'past_cache_size {past_cache_size} != {past[0][0].shape[1]}, passed value wlil be ignored')
                past_cache_size = past[0][0].shape[1]

            indices = jax.lax.dynamic_slice_in_dim(jnp.arange(past_cache_size), past_length, inp_length)

        attention_mask = _make_causal_mask(
            indices,
            past_cache_size=past_cache_size if past_cache_size else None
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
                (layer_past, past_length)
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
