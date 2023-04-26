import haiku as hk
import jax.numpy as jnp
import jmp
from .model import LlamaModel

ALPACA_FORMAT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{}\n\n"
    "### Response:\n"
    "{}"
)

def alpaca_prompt(
    request,
    response_prefix=""
):
    return ALPACA_FORMAT.format(request, response_prefix)


def in_place_tree_map(f, tree):
    if isinstance(tree, dict):
        for k, v in tree.items():
            tree[k] = in_place_tree_map(f, v)
        return tree
    if isinstance(tree, list):
        for i, v in enumerate(tree):
            tree[i] = in_place_tree_map(f, v)
        return tree
    if isinstance(tree, tuple):
        return tuple(in_place_tree_map(f, v) for v in tree)
    if tree is None:
        return None
    return f(tree)

def simple_dtype_policy(dtype=jnp.float16):
    hk.mixed_precision.set_policy(
        LlamaModel,
        jmp.Policy(
            param_dtype=dtype,
            compute_dtype=dtype,
            output_dtype=dtype,
        )
    )
