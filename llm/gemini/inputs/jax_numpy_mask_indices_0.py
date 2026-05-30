
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax.numpy as jnp
import copy

class CallableStr(str):
    def __call__(self, x, k=0):
        if "triu" in self:
            return jnp.triu(x, k)
        else:
            return jnp.tril(x, k)

def mask_indices_inputs():
    list_of_inputs = []

    # Input 1: triu with k=0
    list_of_inputs.append({
        "n": int(3),
        "mask_func": CallableStr("triu"),
        "k": int(0),
        "size": int(6)
    })

    # Input 2: tril with negative k
    list_of_inputs.append({
        "n": int(4),
        "mask_func": CallableStr("tril"),
        "k": int(-1),
        "size": int(3)
    })

    # Input 3: triu with positive k
    list_of_inputs.append({
        "n": int(5),
        "mask_func": CallableStr("triu"),
        "k": int(1),
        "size": int(10)
    })

    # Input 4: tril with k=0
    list_of_inputs.append({
        "n": int(2),
        "mask_func": CallableStr("tril"),
        "k": int(0),
        "size": int(3)
    })

    # Input 5: triu with larger negative k and larger size
    list_of_inputs.append({
        "n": int(6),
        "mask_func": CallableStr("triu"),
        "k": int(-2),
        "size": int(15)
    })

    # Input 6: tril with larger positive k
    list_of_inputs.append({
        "n": int(3),
        "mask_func": CallableStr("tril"),
        "k": int(2),
        "size": int(6)
    })

    # Input 7: triu with larger matrix size
    list_of_inputs.append({
        "n": int(10),
        "mask_func": CallableStr("triu"),
        "k": int(0),
        "size": int(55)
    })

    # Input 8: tril with highly negative k
    list_of_inputs.append({
        "n": int(5),
        "mask_func": CallableStr("tril"),
        "k": int(-3),
        "size": int(3)
    })

    # Input 9: triu with large k
    list_of_inputs.append({
        "n": int(4),
        "mask_func": CallableStr("triu"),
        "k": int(2),
        "size": int(3)
    })

    # Input 10: tril with positive k and medium matrix size
    list_of_inputs.append({
        "n": int(8),
        "mask_func": CallableStr("tril"),
        "k": int(1),
        "size": int(44)
    })

    return list_of_inputs

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
