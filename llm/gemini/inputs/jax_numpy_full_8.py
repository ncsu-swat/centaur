
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_full_8_inputs():
    list_of_inputs = []

    # Input 1, 2D float32 array
    input_dict = {
        "shape": (2, 3),
        "fill_value": np.array(5.0, dtype=np.float32),
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, 1D int32 array with negative value
    input_dict = {
        "shape": (4,),
        "fill_value": np.array(-1, dtype=np.int32),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, 2D boolean array
    input_dict = {
        "shape": (3, 3),
        "fill_value": np.array(True, dtype=np.bool_),
        "dtype": np.dtype('bool')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, 2D float64 array
    input_dict = {
        "shape": (1, 5),
        "fill_value": np.array(3.14159, dtype=np.float64),
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, 3D int64 array
    input_dict = {
        "shape": (2, 2, 2),
        "fill_value": np.array(100, dtype=np.int64),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, broadcastable 1D fill_value to 2D
    input_dict = {
        "shape": (5, 2),
        "fill_value": np.array([3.0, 4.0], dtype=np.float32),
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, exact shape fill_value 2D array
    input_dict = {
        "shape": (3, 1),
        "fill_value": np.array([[10], [20], [30]], dtype=np.int32),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, 2D complex128 array
    input_dict = {
        "shape": (4, 4),
        "fill_value": np.array(2 + 3j, dtype=np.complex128),
        "dtype": np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, 2D int8 array
    input_dict = {
        "shape": (2, 5),
        "fill_value": np.array(-128, dtype=np.int8),
        "dtype": np.dtype('int8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, 3D float16 array
    input_dict = {
        "shape": (1, 1, 1),
        "fill_value": np.array(0.001, dtype=np.float16),
        "dtype": np.dtype('float16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.full_8"] = jax_numpy_full_8_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_8'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_8'], lib="jax", suffix=8)
