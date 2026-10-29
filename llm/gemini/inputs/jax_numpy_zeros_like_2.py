
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, target dtype float32, integer shape
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "dtype": np.dtype("float32"),
        "shape": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, target dtype int64, integer shape
    input_dict = {
        "a": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "dtype": np.dtype("int64"),
        "shape": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, target dtype float64, zero-length integer shape
    input_dict = {
        "a": np.ones((2, 2, 2), dtype=np.float64),
        "dtype": np.dtype("float64"),
        "shape": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array, target dtype bool, integer shape
    input_dict = {
        "a": np.array(42.0, dtype=np.float32),
        "dtype": np.dtype("bool"),
        "shape": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D uint8 array, target dtype uint8, integer shape
    input_dict = {
        "a": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "dtype": np.dtype("uint8"),
        "shape": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array, target dtype complex64, integer shape
    input_dict = {
        "a": np.array([1 + 2j, 3 - 4j], dtype=np.complex64),
        "dtype": np.dtype("complex64"),
        "shape": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, target dtype int16, integer shape
    input_dict = {
        "a": np.zeros((1, 2, 1, 3), dtype=np.float32),
        "dtype": np.dtype("int16"),
        "shape": 8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D bool array, target dtype float16, integer shape
    input_dict = {
        "a": np.array([[True, False], [False, True]], dtype=np.bool_),
        "dtype": np.dtype("float16"),
        "shape": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int8 array with negative values, target dtype int8, integer shape
    input_dict = {
        "a": np.array([-5, -1, 0, 3], dtype=np.int8),
        "dtype": np.dtype("int8"),
        "shape": 7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D complex128 array, target dtype complex128, integer shape
    input_dict = {
        "a": np.ones((2, 1, 4), dtype=np.complex128),
        "dtype": np.dtype("complex128"),
        "shape": 6,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.zeros_like_2"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_2'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_2'], lib="jax", suffix=2)
