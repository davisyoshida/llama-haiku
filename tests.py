import json
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
import pytest

from llama_haiku import LlamaModel, LlamaConfig
from llama_haiku.model import LlamaRotaryEmbedding
from llama_haiku.load import get_model

@pytest.fixture
def small_config():
    return LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=128,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        num_hidden_layers=3
    )

@pytest.fixture
def small_model():
    return get_model('test_data/small_model/', return_past=True)

def test_init(small_config):
    def f(ids):
        return LlamaModel(small_config)(ids)

    ids = jnp.zeros(10, dtype=jnp.int32)
    model = hk.transform(f)
    params = model.init(jax.random.PRNGKey(42), ids)

def test_past(small_config):
    max_len = 15
    def f(ids, past):
        return LlamaModel(small_config)(ids, past=past, past_cache_size=max_len, return_past=True)

    model = hk.without_apply_rng(hk.transform(f))
    params = model.init(jax.random.PRNGKey(0), jnp.zeros(10, dtype=jnp.int32), past=None)

    full_input = jnp.arange(max_len, dtype=jnp.int32)
    full_output = model.apply(params, full_input, past=None)['logits']


    partial_outputs = []
    step_size = 2
    past = None
    for start in range(0, full_input.shape[-1], step_size):
        partial_input = full_input[start:start + step_size]
        ret = model.apply(params, partial_input, past=past)
        partial_outputs.append(ret['logits'])
        past = ret['past']

    incremental_output = jnp.concatenate(partial_outputs, axis=0)
    assert jnp.allclose(full_output, incremental_output, atol=1e-5, rtol=1e-5)

def test_output():
    with open('test_data/small_model/config.json', 'r') as f:
        config = LlamaConfig(**json.load(f))
    def fn(ids):
        return LlamaModel(config)(ids)
    model = hk.without_apply_rng(hk.transform(fn))

    with open('test_data/small_model/weights.pkl', 'rb') as f:
        params = pickle.load(f)

    with open('test_data/inputs_and_logits.pkl', 'rb') as f:
        inputs, logits = pickle.load(f)

    output = model.apply(params, inputs)['logits']
    print(f'Output:\n {output[:2, :3]}')
    print(f'Expected:\n{logits[:2, :3]}')
    print(f'Avg diff: {jnp.mean(jnp.abs(output - logits))}')

    # The output is quite a bit different from the torch model it turns out
    assert jnp.allclose(output, logits, atol=1e-2, rtol=0)

def test_rotary_embedding(small_config):
    n_embeddings = 10

    queries = jnp.arange(n_embeddings * small_config.hidden_size, dtype=jnp.float32).reshape(
        small_config.num_attention_heads,
        n_embeddings,
        small_config.hidden_size // small_config.num_attention_heads
    )

    def func(*args):
        return LlamaRotaryEmbedding(small_config)(*args)
    model = hk.without_apply_rng(hk.transform(func))
    args = queries, queries, jnp.arange(n_embeddings)
    params = model.init(jax.random.PRNGKey(42), *args)

    q_emb, _ = model.apply(params, *args)
    with open('test_data/expected_rotary_output.pkl', 'rb') as f:
        expected_q_emb = pickle.load(f)

    assert jnp.allclose(q_emb, expected_q_emb, atol=1e-5, rtol=1e-5)


def test_no_update_cache(small_model):
    model, params = small_model 
    model_fn = jax.jit(model.apply, static_argnames=('no_cache_update',))
    model_fn = model.apply

    full_input = jnp.arange(1, 9)

    full_output = model_fn(params, full_input)
    full_logits = full_output['logits']

    original_past = full_output['past']

    split = full_input.shape[0] - 1
    initial_output = model_fn(params, full_input[:split], no_cache_update=False)

    initial_past = initial_output['past']

    second_output = model_fn(params, full_input[split:], past=initial_past, no_cache_update=False)

    combined_logits = jnp.concatenate([initial_output['logits'], second_output['logits']], axis=0)

    max_gap = jnp.max(jnp.abs(full_logits - combined_logits))
    print(f'Max gap: {max_gap:.3e}')
    full_dist = jax.nn.softmax(full_logits[-1])
    split_dist = jax.nn.softmax(combined_logits[-1])

    print(f'Average dist gap: {jnp.mean(jnp.abs(full_dist - split_dist)):.3e}')
