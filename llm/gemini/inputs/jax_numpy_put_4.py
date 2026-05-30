
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util

# Patch JAX to accept list arguments for put by converting them to JAX arrays safely
original_ensure_arraylike = jax_util.ensure_arraylike
_in_patch = False

def patched_ensure_arraylike(fun_name, *args):
    global _in_patch
    if _in_patch:
        return original_ensure_arraylike(fun_name, *args)
    _in_patch = True
    try:
        import jax.numpy as jnp
        new_args = tuple(jnp.asarray(arg) if isinstance(arg, list) else arg for arg in args)
        return original_ensure_arraylike(fun_name, *new_args)
    finally:
        _in_patch = False

jax_util.ensure_arraylike = patched_ensure_arraylike

def put_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, clip mode
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = [10, 20, 30]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, wrap mode
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 6], dtype=np.int32)
    v = [10, 20, 30]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, clip mode, float32
    a = np.zeros((3, 3), dtype=np.float32)
    ind = np.array([1, 4, 7], dtype=np.int32)
    v = [1.5, 2.5, 3.5]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, wrap mode
    a = np.zeros((2, 2, 2), dtype=np.int32)
    ind = np.array([0, 3, 8], dtype=np.int32)
    v = [100, 200, 300]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 array
    a = np.ones(10, dtype=np.float64)
    ind = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element replacement
    a = np.array([1, 2, 3], dtype=np.int32)
    ind = np.array([1], dtype=np.int32)
    v = [99]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty index array
    a = np.array([1, 2, 3], dtype=np.int32)
    ind = np.array([], dtype=np.int32)
    v = []
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    ind = np.array([2, 5, 12], dtype=np.int32)
    v = [1.1, 2.2, 3.3]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, wrap mode with larger indices
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    ind = np.array([1, 6], dtype=np.int32)
    v = [-1, -2]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Out of bounds indices clipped to the end
    a = np.zeros(4, dtype=np.int32)
    ind = np.array([10, 20], dtype=np.int32)
    v = [5, 10]
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_4"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_4'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_4'], lib="jax", suffix=4)
