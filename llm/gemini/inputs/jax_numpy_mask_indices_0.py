
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp

class CallableStr(str):
    def __call__(self, x, k=0):
        if "triu" in self:
            return jnp.triu(x, k)
        return jnp.tril(x, k)

def mask_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic triu on 3x3 matrix
    list_of_inputs.append({
        "n": 3,
        "mask_func": CallableStr("triu"),
        "k": 0,
        "size": 6
    })

    # Input 2: Basic tril on 3x3 matrix
    list_of_inputs.append({
        "n": 3,
        "mask_func": CallableStr("tril"),
        "k": 0,
        "size": 6
    })

    # Input 3: Positive offset k with triu
    list_of_inputs.append({
        "n": 4,
        "mask_func": CallableStr("triu"),
        "k": 1,
        "size": 6
    })

    # Input 4: Negative offset k with tril
    list_of_inputs.append({
        "n": 4,
        "mask_func": CallableStr("tril"),
        "k": -1,
        "size": 6
    })

    # Input 5: Negative offset k with triu
    list_of_inputs.append({
        "n": 5,
        "mask_func": CallableStr("triu"),
        "k": -2,
        "size": 15
    })

    # Input 6: Positive offset k with tril
    list_of_inputs.append({
        "n": 5,
        "mask_func": CallableStr("tril"),
        "k": 2,
        "size": 15
    })

    # Input 7: Small matrix n=2
    list_of_inputs.append({
        "n": 2,
        "mask_func": CallableStr("triu"),
        "k": 0,
        "size": 3
    })

    # Input 8: Minimal 1x1 matrix
    list_of_inputs.append({
        "n": 1,
        "mask_func": CallableStr("tril"),
        "k": 0,
        "size": 1
    })

    # Input 9: Large offset with triu
    list_of_inputs.append({
        "n": 6,
        "mask_func": CallableStr("triu"),
        "k": 3,
        "size": 6
    })

    # Input 10: Standard tril on 4x4 matrix
    list_of_inputs.append({
        "n": 4,
        "mask_func": CallableStr("tril"),
        "k": 0,
        "size": 10
    })

    # Input 11: Empty matrix
    list_of_inputs.append({
        "n": 0,
        "mask_func": CallableStr("triu"),
        "k": 0,
        "size": 0
    })

    # Input 12: Positive offset with tril
    list_of_inputs.append({
        "n": 3,
        "mask_func": CallableStr("tril"),
        "k": 1,
        "size": 8
    })

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.mask_indices"] = mask_indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mask_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mask_indices'.")


check_valid('jax.numpy.mask_indices', generated_inputs['jax.numpy.mask_indices'], lib="jax", suffix=0)
