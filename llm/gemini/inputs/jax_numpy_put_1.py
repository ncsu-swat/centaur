
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_put_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float array, in-bounds indices, clip mode
    a = np.zeros(5, dtype=np.float32)
    ind = np.array([1, 3], dtype=np.int32)
    v = np.array([1.5, 2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int array, out-of-bounds indices, wrap mode
    a = np.arange(10, dtype=np.int32)
    ind = np.array([2, 5, 12], dtype=np.int32)
    v = np.array([100, 200, 300], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array (flattened indexing), clip mode
    a = np.ones((3, 3), dtype=np.float64)
    ind = np.array([0, 4, 8], dtype=np.int32)
    v = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, 2D index array, wrap mode
    a = np.zeros((4, 4), dtype=np.int32)
    ind = np.array([[0, 5], [10, 15]], dtype=np.int32)
    v = np.array([[9, 8], [7, 6]], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float16 array, wrap mode
    a = np.random.randn(8).astype(np.float16)
    ind = np.array([1, 3, 5], dtype=np.int32)
    v = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, clip mode
    a = np.zeros((2, 2, 2), dtype=np.int32)
    ind = np.array([1, 6], dtype=np.int32)
    v = np.array([42, 43], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array, clip mode
    a = np.zeros(6, dtype=bool)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = np.array([True, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with matching shape values, wrap mode
    a = np.arange(5, dtype=np.float32)
    ind = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    v = np.array([9.9, 9.9, 9.9, 9.9, 9.9], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large indices, clipping to the end
    a = np.zeros(3, dtype=np.int32)
    ind = np.array([10, 20], dtype=np.int32)
    v = np.array([5, 10], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, flattened indices, clip mode
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    ind = np.array([0, 7, 15], dtype=np.int32)
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_1"] = jax_numpy_put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_1'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_1'], lib="jax", suffix=1)
