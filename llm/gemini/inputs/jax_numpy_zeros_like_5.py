
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, override with float32 and shape 5
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative elements, override with int32 and shape 8
    a = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "dtype": "int32",
        "shape": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, override with float64 and shape 2
    a = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "dtype": "float64",
        "shape": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array, override with bool and shape 10
    a = np.random.choice([True, False], size=6)
    input_dict = {
        "a": a,
        "dtype": "bool",
        "shape": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, override with int16 and shape 3
    a = np.random.randn(2, 3, 1, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "int16",
        "shape": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, override with float32 and shape 0 (empty array)
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "float32",
        "shape": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64 array with negatives, override with int64 and shape 15
    a = np.random.randint(-100, 100, size=5).astype(np.int64)
    input_dict = {
        "a": a,
        "dtype": "int64",
        "shape": 15
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int8 array, override with float16 and shape 6
    a = np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int8)
    input_dict = {
        "a": a,
        "dtype": "float16",
        "shape": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D uint8 array, override with uint8 and shape 12
    a = np.random.randint(0, 255, size=(4, 2)).astype(np.uint8)
    input_dict = {
        "a": a,
        "dtype": "uint8",
        "shape": 12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, override with complex64 and shape 4
    a = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "dtype": "complex64",
        "shape": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_like_5"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_5'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_5'], lib="jax", suffix=5)
