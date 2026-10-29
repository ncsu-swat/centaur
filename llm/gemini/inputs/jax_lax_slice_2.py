
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def slice_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, stride 1
    input_dict = {
        "operand": np.arange(10, dtype=np.int32),
        "start_indices": (2,),
        "limit_indices": (8,),
        "strides": (1,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, mixed strides
    input_dict = {
        "operand": np.ones((5, 6), dtype=np.float32),
        "start_indices": (1, 0),
        "limit_indices": (4, 5),
        "strides": (1, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array
    input_dict = {
        "operand": np.zeros((4, 4, 4), dtype=np.float64),
        "start_indices": (0, 1, 2),
        "limit_indices": (3, 4, 3),
        "strides": (2, 1, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array with stride 2
    input_dict = {
        "operand": np.array([True, False, True, True, False], dtype=np.bool_),
        "start_indices": (0,),
        "limit_indices": (5,),
        "strides": (2,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int64 array, full slice
    input_dict = {
        "operand": np.ones((2, 3, 4, 5), dtype=np.int64),
        "start_indices": (0, 0, 0, 0),
        "limit_indices": (2, 3, 4, 5),
        "strides": (1, 1, 1, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array with stride 2 in both dimensions
    input_dict = {
        "operand": np.zeros((6, 8), dtype=np.complex64),
        "start_indices": (1, 2),
        "limit_indices": (5, 7),
        "strides": (2, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, empty result (start == limit)
    input_dict = {
        "operand": np.arange(20, dtype=np.float32),
        "start_indices": (5,),
        "limit_indices": (5,),
        "strides": (1,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D uint8 array with different strides
    input_dict = {
        "operand": np.ones((10, 10, 10), dtype=np.uint8),
        "start_indices": (2, 3, 4),
        "limit_indices": (8, 9, 10),
        "strides": (3, 2, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float16 array with stride 3 along columns
    input_dict = {
        "operand": np.arange(24, dtype=np.float16).reshape(4, 6),
        "start_indices": (0, 1),
        "limit_indices": (4, 6),
        "strides": (1, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int16 array
    input_dict = {
        "operand": np.ones((2, 2, 2, 2, 2), dtype=np.int16),
        "start_indices": (0, 1, 0, 1, 0),
        "limit_indices": (1, 2, 2, 2, 1),
        "strides": (1, 1, 1, 1, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int32 array with large stride
    input_dict = {
        "operand": np.arange(100, dtype=np.int32),
        "start_indices": (10,),
        "limit_indices": (90,),
        "strides": (15,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.slice_2"] = slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_2'.")


check_valid('jax.lax.slice', generated_inputs['jax.lax.slice_2'], lib="jax", suffix=2)
