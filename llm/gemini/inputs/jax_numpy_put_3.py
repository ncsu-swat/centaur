
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

_original_put = jnp.put

def patched_put(a, ind, v, mode=None, *, inplace=True):
    if isinstance(ind, tuple):
        ind = np.array(ind, dtype=np.int32)
    return _original_put(a, ind, v, mode=mode, inplace=inplace)

jnp.put = patched_put

def jax_numpy_put_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, normal indices, clip mode
    a = np.zeros(5, dtype=np.int32)
    ind = (0, 2, 4)
    v = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array, wrap mode with out-of-bound indices
    a = np.ones(10, dtype=np.float32)
    ind = (1, 3, 5, 12)
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int array, referencing flattened indices, clip mode
    a = np.zeros((3, 3), dtype=np.int32)
    ind = (1, 4, 7)
    v = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, single index
    a = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
    ind = (5,)
    v = np.array([99.9], dtype=np.float64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted value (1-element value array to multiple indices)
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ind = (0, 1, 2, 3, 4)
    v = np.array([100], dtype=np.int64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Normal positive indices
    a = np.zeros(6, dtype=np.int32)
    ind = (5, 3)
    v = np.array([10, 20], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large out-of-bound indices with clip mode
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ind = (10, 20)
    v = np.array([100.0, 200.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large out-of-bound indices with wrap mode
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ind = (10, 20)
    v = np.array([100.0, 200.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, clip mode
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    ind = (0, 5, 10, 15)
    v = np.array([9, 9, 9, 9], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array updates
    a = np.zeros(4, dtype=bool)
    ind = (1, 3)
    v = np.array([True, True], dtype=bool)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_3"] = jax_numpy_put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_3'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_3'], lib="jax", suffix=3)
