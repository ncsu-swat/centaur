
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, overridden with float32 and different 1D shape
    a = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, overridden with int32 and different 2D shape
    a = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "dtype": "int32",
        "shape": (2, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, overridden with float64 and different 3D shape
    a = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "dtype": "float64",
        "shape": (3, 3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D bool array, overridden with bool and 2D shape
    a = np.array([True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "dtype": "bool",
        "shape": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int16 array, overridden with float32 and 1D shape
    a = np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.int16)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": (8,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array, overridden with complex64 and same shape
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "dtype": "complex64",
        "shape": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D array (scalar), overridden with int64 and 2D shape
    a = np.array(42).astype(np.int32)
    input_dict = {
        "a": a,
        "dtype": "int64",
        "shape": (1, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float16 array, overridden with float16 and 3D shape
    a = np.random.randn(1, 1, 1, 1, 5).astype(np.float16)
    input_dict = {
        "a": a,
        "dtype": "float16",
        "shape": (2, 2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D uint8 array, overridden with uint8 and 1D shape
    a = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    input_dict = {
        "a": a,
        "dtype": "uint8",
        "shape": (16,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int8 array, overridden with int32 and 4D shape
    a = np.random.randint(-100, 100, size=(1, 2, 3)).astype(np.int8)
    input_dict = {
        "a": a,
        "dtype": "int32",
        "shape": (1, 2, 3, 4)
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
