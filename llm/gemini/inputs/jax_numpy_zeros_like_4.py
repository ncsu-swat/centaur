
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with shape override to 2D int32
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "dtype": "int32",
        "shape": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array with 1D bool override
    input_dict = {
        "a": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "dtype": "bool",
        "shape": (4,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float array overridden with 3D float64
    input_dict = {
        "a": np.ones((2, 3, 4), dtype=np.float32),
        "dtype": "float64",
        "shape": (3, 2, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array overridden with 2D float32
    input_dict = {
        "a": np.array(5.0, dtype=np.float32),
        "dtype": "float32",
        "shape": (2, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex array with complex64 dtype and scalar tuple shape
    input_dict = {
        "a": np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        "dtype": "complex64",
        "shape": (),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float array with same shape and uint8 dtype
    input_dict = {
        "a": np.random.randn(2, 2, 3, 3).astype(np.float32),
        "dtype": "uint8",
        "shape": (2, 2, 3, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array overridden with int64 and 1D shape
    input_dict = {
        "a": np.array([[True, False], [False, True]], dtype=np.bool_),
        "dtype": "int64",
        "shape": (10,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative float values, reshaped to a higher-dimensional shape
    input_dict = {
        "a": np.array([-1.5, -2.5, -3.5], dtype=np.float32),
        "dtype": "float32",
        "shape": (1, 3, 1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer array overridden with int16 and empty shape
    input_dict = {
        "a": np.arange(12, dtype=np.int64).reshape(3, 4),
        "dtype": "int16",
        "shape": (0,),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with 5D shape and float16 dtype
    input_dict = {
        "a": np.zeros((3, 3), dtype=np.float32),
        "dtype": "float16",
        "shape": (1, 2, 1, 3, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex128 array with int8 dtype
    input_dict = {
        "a": np.array([[-1 - 1j]], dtype=np.complex128),
        "dtype": "int8",
        "shape": (5, 5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.zeros_like_4"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_4'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_4'], lib="jax", suffix=4)
