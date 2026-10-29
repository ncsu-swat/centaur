
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int array with mode 'clip'
    a = np.zeros(5, dtype=np.int32)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with out-of-bounds indices and mode 'wrap'
    a = np.zeros(5, dtype=np.float32)
    ind = np.array([0, 2, 6], dtype=np.int32)
    v = np.array([10.5, 20.5, 30.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, flattened indexing with mode 'clip'
    a = np.ones((3, 4), dtype=np.float64)
    ind = np.array([1, 5, 11], dtype=np.int64)
    v = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with negative indices and mode 'clip'
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    ind = np.array([-1, -5, 2], dtype=np.int32)
    v = np.array([99, 100, 101], dtype=np.int64)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean array with mode 'wrap'
    a = np.array([True, False, True, False, False], dtype=np.bool_)
    ind = np.array([1, 3], dtype=np.int32)
    v = np.array([True, True], dtype=np.bool_)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32 array with repeated indices and mode 'clip'
    a = np.zeros(6, dtype=np.float32)
    ind = np.array([0, 3, 3], dtype=np.int32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty indices and values with mode 'clip'
    a = np.arange(10, dtype=np.float32)
    ind = np.array([], dtype=np.int32)
    v = np.array([], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional index array with mode 'wrap'
    a = np.zeros((4, 4), dtype=np.int32)
    ind = np.array([[0, 2], [5, 15]], dtype=np.int32)
    v = np.array([[7, 8], [9, 10]], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array with single element update and mode 'clip'
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    ind = np.array([7], dtype=np.int32)
    v = np.array([42.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 array with out-of-bounds indices and mode 'wrap'
    a = np.array([[10, 20], [30, 40]], dtype=np.int32)
    ind = np.array([4, 5, 8], dtype=np.int32)
    v = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.put_1"] = put_inputs()

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
