
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_like_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D
    input_dict = {
        "arr": np.array([1, 2, 3], dtype=np.int32),
        "like_arr": np.zeros((2, 3), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar to 3D
    input_dict = {
        "arr": np.array(4.5, dtype=np.float32),
        "like_arr": np.ones((2, 3, 4), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: (1, 4) to (3, 4)
    input_dict = {
        "arr": np.array([[1, 2, 3, 4]], dtype=np.int32),
        "like_arr": np.zeros((3, 4), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: (3, 1) to (3, 4)
    input_dict = {
        "arr": np.array([[1], [2], [3]], dtype=np.float32),
        "like_arr": np.zeros((3, 4), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same shape 1D
    input_dict = {
        "arr": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        "like_arr": np.ones((5,), dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: (3, 4) to (2, 3, 4)
    input_dict = {
        "arr": np.random.randn(3, 4).astype(np.float32),
        "like_arr": np.zeros((2, 3, 4), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: (1, 1) to (5, 5) float64
    input_dict = {
        "arr": np.array([[3.14]], dtype=np.float64),
        "like_arr": np.zeros((5, 5), dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: boolean array
    input_dict = {
        "arr": np.array([True], dtype=np.bool_),
        "like_arr": np.zeros((2, 2), dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative values in int32
    input_dict = {
        "arr": np.array([-1, -2, -3], dtype=np.int32),
        "like_arr": np.zeros((4, 3), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: high dimensional (1, 5, 1) to (2, 5, 3)
    input_dict = {
        "arr": np.random.randn(1, 5, 1).astype(np.float32),
        "like_arr": np.zeros((2, 5, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.broadcast_like"] = broadcast_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_like'.")


check_valid('jax.lax.broadcast_like', generated_inputs['jax.lax.broadcast_like'], lib="jax", suffix=0)
