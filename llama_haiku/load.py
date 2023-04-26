import json
import math
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import haiku as hk

from .model import LlamaConfig, LlamaModel
from .utils import in_place_tree_map

def load_config(path, name='config.json'):
    path = Path(path)
    with (path / name).open() as f:
        return LlamaConfig(**json.load(f))

def load_weights(path, name='weights.pkl'):
    path = Path(path)
    with (path / name).open('rb') as f:
        params = pickle.load(f)
    gpu = jax.devices('gpu')[0]
    return in_place_tree_map(lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.float16), gpu), params)

def get_model(model_dir, return_past=False, return_hidden=False):
    model_dir = Path(model_dir)
    config = load_config(model_dir)
    params = load_weights(model_dir)

    def fn(input_ids, use_cache_size=None, past=None):
        model = LlamaModel(config)
        ret = model(input_ids, past=past, past_cache_size=use_cache_size, return_past=return_past, return_hidden=return_hidden)
        return ret

    model = hk.without_apply_rng(hk.transform(fn))
    return model, params

def get_generator(model_dir, cache_step_size=25, donate_past=True, return_hidden=False, output_transform=None):
    model, params = get_model(model_dir, return_past=True, return_hidden=return_hidden)


    def model_fn(params, input_ids, use_cache_size, past):
        ret = model.apply(params, input_ids, use_cache_size, past)
        if output_transform is not None:
            ret = output_transform(ret)
        return ret

    donate_argnums = (3,) if donate_past else ()
    jit_fn = jax.jit(model_fn, static_argnums=(2,), donate_argnums=donate_argnums)


    def step_fn(input_ids, past=None):
        input_length = input_ids.shape[-1]
        total_length = input_length + (past[1] if past else 0)

        use_cache_size = math.ceil(total_length / cache_step_size) * cache_step_size

        if past:
            curr_cache_size = past[0][0][0].shape[-2]
            if curr_cache_size < use_cache_size:
                curr_cache, length = past
                new_cache = jax.tree_map(
                    lambda x: jnp.pad(x, ((0, 0), (0, use_cache_size - curr_cache_size), (0, 0))),
                    curr_cache
                )
                past = (new_cache, length)

        return jit_fn(params, input_ids, use_cache_size, past)

    return step_fn

if __name__ == '__main__':
    main()
