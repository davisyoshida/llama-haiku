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

def load_config(path, name='config.json', overwrite_config_vals=None):
    path = Path(path)

    with (path / name).open() as f:
        conf_dict = json.load(f)
    if overwrite_config_vals is not None:
        conf_dict.update(overwrite_config_vals)

    return LlamaConfig(**conf_dict)

def load_weights(path, name='weights.pkl', device=None):
    path = Path(path)
    with (jax.default_device(device) if device is not None else nullcontext()):
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

def get_model(
    model_dir,
    return_past=False,
    return_hidden=False,
    device=None,
    custom_getter=None,
    get_params=True,
    overwrite_config_vals=None,
    return_config=False
):
    model_dir = Path(model_dir)
    config = load_config(model_dir, overwrite_config_vals=overwrite_config_vals)

    params = load_weights(model_dir, device=device) if get_params else None

    def fn(
        input_ids,
        past=None,
        use_cache_size=None,
        do_checkpoint=False,
        do_mlp_checkpoint=False,
        mlp_block_size=None,
        use_flash_attention=False,
        no_cache_update=False
    ):
        with hk.custom_getter(custom_getter) if custom_getter is not None else nullcontext():
            model = LlamaModel(config)
            ret = model(
                input_ids,
                past=past,
                past_cache_size=use_cache_size,
                return_past=return_past,
                return_hidden=return_hidden,
                checkpoint=do_checkpoint,
                checkpoint_mlp=do_mlp_checkpoint,
                mlp_block_size=mlp_block_size,
                no_cache_update=no_cache_update
            )
        return ret

    model = hk.without_apply_rng(hk.transform(fn))
    ret = (model, params)
    if return_config:
        ret += (config,)

    return ret



def get_generator(
    model_dir,
    cache_step_size=25,
    donate_past=True,
    apply_wrapper=None,
    return_hidden=False,
    params_wrapper=None,
    device=None,
    overwrite_config_vals=None,
    use_flash_attention=False
):
    model, params = get_model(
        model_dir,
        return_past=True,
        return_hidden=return_hidden,
        device=device,
        overwrite_config_vals=overwrite_config_vals
    )

    apply_fn = model.apply
    if apply_wrapper is not None:
        apply_fn = apply_wrapper(apply_fn)
    if params_wrapper is not None:
        params = params_wrapper(params)

    def model_fn(params, input_ids, use_cache_size, past):
        ret = apply_fn(params, input_ids, use_cache_size, past, use_flash_attention=use_flash_attention)
        return ret

    donate_argnums = (2,) if donate_past else ()
    jit_fn = jax.jit(model_fn, static_argnums=(3,), donate_argnums=donate_argnums)

    def step_fn(input_ids, past=None):
        input_length = input_ids.shape[-1]
        total_length = input_length + (past[1] if past else 0)

        use_cache_size = math.ceil(total_length / cache_step_size) * cache_step_size

        added_padding = 0
        if past:
            curr_cache_size = past[0][0][0].shape[-2]
            if curr_cache_size < use_cache_size:
                curr_cache, length = past
                new_cache = jax.tree_map(
                    lambda x: jnp.pad(x, ((0, 0), (0, use_cache_size - curr_cache_size), (0, 0))),
                    curr_cache
                )
                past = (new_cache, length)
        else:
            pad_to = np.ceil(input_length / cache_step_size).astype(int) * cache_step_size
            added_padding = pad_to - input_length
            if added_padding > 0:
                input_ids = jnp.pad(input_ids, ((0, added_padding)), constant_values=0)

        result = jit_fn(params, input_ids, past, use_cache_size)

        if added_padding > 0:
            result['logits'] = result['logits'][:-added_padding, :]
            past, past_length = result['past']

            past_length -= added_padding
            result['past'] = (past, past_length)

            if 'hidden' in result:
                result['hidden'] = jax.tree_map(lambda h: h[:-added_padding, :], result['hidden'])
        return result

    return step_fn
