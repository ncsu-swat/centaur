
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with clip mode
    input_dict = {
        "a": np.zeros(5, dtype=np.int32),
        "ind": np.array([0, 2, 4], dtype=np.int32),
        "v": np.array([10, 20, 30], dtype=np.int32),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with wrap mode
    input_dict = {
        "a": np.zeros(5, dtype=np.float32),
        "ind": np.array([1, 6], dtype=np.int32),
        "v": np.array([1.5, 3.5], dtype=np.float32),
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array (indices over flattened array)
    input_dict = {
        "a": np.zeros((3, 4), dtype=np.int64),
        "ind": np.array([0, 5, 11], dtype=np.int32),
        "v": np.array([100, 200, 300], dtype=np.int64),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array with wrap mode
    input_dict = {
        "a": np.ones((2, 3), dtype=np.float64),
        "ind": np.array([7, 8], dtype=np.int32),
        "v": np.array([-0.5, -1.5], dtype=np.float64),
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array with clip mode
    input_dict = {
        "a": np.zeros((2, 2, 3), dtype=np.float32),
        "ind": np.array([0, 3, 7, 11], dtype=np.int32),
        "v": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D boolean array
    input_dict = {
        "a": np.zeros(6, dtype=bool),
        "ind": np.array([1, 3, 5], dtype=np.int32),
        "v": np.array([True, True, True], dtype=bool),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with single element update
    input_dict = {
        "a": np.arange(10, dtype=np.int32),
        "ind": np.array([4], dtype=np.int32),
        "v": np.array([999], dtype=np.int32),
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D integer array with wrap mode
    input_dict = {
        "a": np.zeros((3, 3, 3), dtype=np.int32),
        "ind": np.array([27, 28, 29], dtype=np.int32),
        "v": np.array([7, 8, 9], dtype=np.int32),
        "mode": "wrap",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float array with clip mode
    input_dict = {
        "a": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "ind": np.array([0, 15, 20], dtype=np.int32),
        "v": np.array([11.0, 22.0, 33.0], dtype=np.float32),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with 2D index array
    input_dict = {
        "a": np.zeros((4, 4), dtype=np.float64),
        "ind": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "v": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "mode": "clip",
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.put_5"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_5'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_5'], lib="jax", suffix=5)
