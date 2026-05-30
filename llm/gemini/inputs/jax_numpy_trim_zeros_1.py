
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trim_zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, trimming both ends
    filt = np.array([0, 0, 1, 2, 3, 0, 0], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array, trimming leading zeros only
    filt = np.array([0.0, 0.0, -1.5, 2.3, 0.0], dtype=np.float32)
    input_dict = {
        "filt": filt,
        "trim": "f",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int64 array, trimming trailing zeros only
    filt = np.array([5, 6, 7, 0, 0, 0], dtype=np.int64)
    input_dict = {
        "filt": filt,
        "trim": "b",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, trimming along axis 0 (rows)
    filt = np.array([
        [0, 0, 0],
        [1, 2, 3],
        [4, 5, 6],
        [0, 0, 0]
    ], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, trimming trailing zeros along axis 1 (columns)
    filt = np.array([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [5, 6, 0, 0]
    ], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "b",
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float array, trimming leading zeros along axis 1 (columns)
    filt = np.array([
        [0.0, 0.0, 1.1, 2.2],
        [0.0, 0.0, 3.3, 4.4]
    ], dtype=np.float64)
    input_dict = {
        "filt": filt,
        "trim": "f",
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, trimming along axis 0
    filt = np.zeros((4, 2, 2), dtype=np.int32)
    filt[1:3, :, :] = 1
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, trimming along axis 1
    filt = np.zeros((2, 4, 2), dtype=np.int32)
    filt[:, 1:3, :] = 5
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, trimming along axis 2
    filt = np.zeros((2, 2, 5), dtype=np.float32)
    filt[:, :, 1:4] = -2.5
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array (interpreting False as zero)
    filt = np.array([False, False, True, True, False], dtype=bool)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with negative values, trimming trailing zeros along axis 0
    filt = np.array([
        [-1, -2, -3],
        [-4, -5, -6],
        [0, 0, 0]
    ], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "b",
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trim_zeros_1"] = trim_zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trim_zeros_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trim_zeros_1'.")


check_valid('jax.numpy.trim_zeros', generated_inputs['jax.numpy.trim_zeros_1'], lib="jax", suffix=1)
