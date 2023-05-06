from contextlib import nullcontext
import json
import math
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .model import LlamaConfig, LlamaModel
from .utils import simple_dtype_policy

def load_config(path, name='config.json'):
    path = Path(path)
    with (path / name).open() as f:
        return LlamaConfig(**json.load(f))

def load_weights(path, name='weights.pkl', device=None):
    path = Path(path)
    with (path / name).open('rb') as f:
        params = pickle.load(f)
    if device is None:
        device = jax.devices('gpu')[0]

    def cast_and_put(numpy_x):
        dtype = None
        if numpy_x.dtype == np.float32:
            dtype = jnp.float16

        x = jnp.asarray(numpy_x, dtype=dtype)
        x = jax.device_put(x, device)

        return x

    params, structure = jax.tree_util.tree_flatten(params)
    mapped_params = []
    while params:
        mapped_params.append(cast_and_put(params.pop()))
    mapped_params.reverse()
    return jax.tree_util.tree_unflatten(structure, mapped_params)

def get_model(model_dir, lora_file=None, return_past=False, return_hidden=False, device=None, custom_getter=None):
    model_dir = Path(model_dir)
    config = load_config(model_dir)
    params = load_weights(model_dir, device=device)

    simple_dtype_policy()

    def fn(input_ids, use_cache_size=None, past=None, do_checkpoint=False):
        with hk.custom_getter(custom_getter) if custom_getter is not None else nullcontext():
            model = LlamaModel(config)
            ret = model(input_ids, past=past, past_cache_size=use_cache_size, return_past=return_past, return_hidden=return_hidden, checkpoint=do_checkpoint)
        return ret

    model = hk.without_apply_rng(hk.transform(fn))
    return model, params

def get_generator(
    model_dir,
    cache_step_size=25,
    donate_past=True,
    apply_wrapper=None,
    return_hidden=False,
    params_wrapper=None,
):
    model, params = get_model(
        model_dir,
        return_past=True,
        return_hidden=return_hidden,
    )

    apply_fn = model.apply
    if apply_wrapper is not None:
        apply_fn = apply_wrapper(apply_fn)
    if params_wrapper is not None:
        params = params_wrapper(params)

    def model_fn(params, input_ids, use_cache_size, past):
        ret = apply_fn(params, input_ids, use_cache_size, past)
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
