
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive float fill_value, float32 dtype, 1D shape
    input_dict = {
        "a": np.zeros(5, dtype=np.float32),
        "fill_value": 3.14,
        "dtype": np.dtype(np.float32),
        "shape": (5,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, negative float fill_value, int32 dtype, 2D shape
    input_dict = {
        "a": np.ones((2, 3), dtype=np.int32),
        "fill_value": -7.0,
        "dtype": np.dtype(np.int32),
        "shape": (3, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, zero float fill_value, float64 dtype, 3D shape
    input_dict = {
        "a": np.arange(8, dtype=np.float64).reshape((2, 2, 2)),
        "fill_value": 0.0,
        "dtype": np.dtype(np.float64),
        "shape": (2, 2, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array, positive float fill_value, bool dtype, 1D shape
    input_dict = {
        "a": np.array(1.0, dtype=np.float32),
        "fill_value": 1.0,
        "dtype": np.dtype(np.bool_),
        "shape": (4,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, fractional float fill_value, float16 dtype, 4D shape
    input_dict = {
        "a": np.ones((1, 2, 1, 3), dtype=np.float16),
        "fill_value": 0.5,
        "dtype": np.dtype(np.float16),
        "shape": (1, 2, 3, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, integer-like float fill_value, uint8 dtype, 2D shape
    input_dict = {
        "a": np.full((3, 3), 5, dtype=np.uint8),
        "fill_value": 255.0,
        "dtype": np.dtype(np.uint8),
        "shape": (2, 5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, negative float fill_value, complex64 dtype, 2D shape
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.complex64),
        "fill_value": -2.5,
        "dtype": np.dtype(np.complex64),
        "shape": (3, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, float fill_value, int64 dtype, 0D shape (scalar)
    input_dict = {
        "a": np.zeros((4, 4), dtype=np.int64),
        "fill_value": 42.0,
        "dtype": np.dtype(np.int64),
        "shape": (),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, float fill_value, float32 dtype, 5D shape
    input_dict = {
        "a": np.empty((1, 1, 1, 1, 1), dtype=np.float32),
        "fill_value": 99.9,
        "dtype": np.dtype(np.float32),
        "shape": (1, 2, 1, 2, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, negative float fill_value, int16 dtype, 3D shape
    input_dict = {
        "a": np.ones((2, 1, 2), dtype=np.int16),
        "fill_value": -128.0,
        "dtype": np.dtype(np.int16),
        "shape": (4, 1, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array, float fill_value, complex128 dtype, 1D shape
    input_dict = {
        "a": np.zeros(2, dtype=np.complex128),
        "fill_value": 1.414,
        "dtype": np.dtype(np.complex128),
        "shape": (10,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_like_2"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_2'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_2'], lib="jax", suffix=2)
