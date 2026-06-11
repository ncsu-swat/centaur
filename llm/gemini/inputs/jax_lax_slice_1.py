
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, whole slice
    operand = np.arange(10).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [0],
        "limit_indices": [10],
        "strides": [1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, sliced with stride 2
    operand = np.arange(20).astype(np.int32)
    input_dict = {
        "operand": operand,
        "start_indices": [2],
        "limit_indices": [18],
        "strides": [2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, subset slice
    operand = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [1, 0],
        "limit_indices": [3, 2],
        "strides": [1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, strided slice
    operand = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "operand": operand,
        "start_indices": [0, 0],
        "limit_indices": [6, 6],
        "strides": [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [0, 1, 1],
        "limit_indices": [2, 3, 3],
        "strides": [1, 1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 array with non-unit strides
    operand = np.random.randint(-10, 10, size=(5, 5, 5)).astype(np.int64)
    input_dict = {
        "operand": operand,
        "start_indices": [0, 1, 0],
        "limit_indices": [5, 4, 5],
        "strides": [2, 1, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [0, 0, 0, 0],
        "limit_indices": [2, 1, 2, 1],
        "strides": [1, 1, 1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array (Scalar) slice
    operand = np.array(42.0).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [],
        "limit_indices": [],
        "strides": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array with slice
    operand = np.random.randn(1000).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_indices": [100],
        "limit_indices": [500],
        "strides": [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D boolean array slice
    operand = (np.random.randn(4, 4) > 0).astype(np.bool_)
    input_dict = {
        "operand": operand,
        "start_indices": [1, 1],
        "limit_indices": [3, 4],
        "strides": [1, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.slice_1"] = slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_1'.")


check_valid('jax.lax.slice', generated_inputs['jax.lax.slice_1'], lib="jax", suffix=1)
