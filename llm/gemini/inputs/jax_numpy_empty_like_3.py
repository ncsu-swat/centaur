
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 prototype, reshaped to 1D
    prototype = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.float32),
        "shape": [4],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 prototype, reshaped to 2D
    prototype = np.arange(6, dtype=np.int32).reshape(2, 3)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.int32),
        "shape": [3, 5],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 prototype, reshaped to 3D
    prototype = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.float32),
        "shape": [2, 2, 2],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array prototype, bool dtype, reshaped to 1D
    prototype = np.array(True, dtype=np.bool_)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.bool_),
        "shape": [1],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D complex64 prototype, reshaped to 4D
    prototype = np.zeros((1, 2, 2, 1), dtype=np.complex64)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.complex64),
        "shape": [1, 2, 3, 4],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 prototype, reshaped with zero dimension
    prototype = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.int32),
        "shape": [0],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D bool prototype, reshaped to 2D
    prototype = np.array([True, False, True], dtype=np.bool_)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.bool_),
        "shape": [5, 5],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float16 prototype, reshaped to 1D
    prototype = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float16)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.float16),
        "shape": [6],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 prototype, reshaped to 3D with zero dimension
    prototype = np.ones((2, 3, 2), dtype=np.int32)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.int32),
        "shape": [2, 0, 3],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex64 prototype, reshaped to 3D
    prototype = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.complex64),
        "shape": [2, 3, 4],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float32 prototype, reshaped to 1D
    prototype = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "prototype": prototype,
        "dtype": np.dtype(np.float32),
        "shape": [9],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.empty_like_3"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_3'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_3'], lib="jax", suffix=3)
