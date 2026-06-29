
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def max_inputs():
    list_of_inputs = []

    # Input 1: Scalar integers (Python ints)
    input_dict = {
        "x": 5,
        "y": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0-D numpy arrays (int32)
    input_dict = {
        "x": np.array(42, dtype=np.int32),
        "y": np.array(-10, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1-D numpy arrays (int32)
    input_dict = {
        "x": np.array([1, -2, 3], dtype=np.int32),
        "y": np.array([0, 5, -3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2-D numpy arrays (int64)
    input_dict = {
        "x": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "y": np.array([[4, 3], [2, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3-D numpy arrays (int16)
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.int16) * 5,
        "y": np.ones((2, 2, 2), dtype=np.int16) * 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with same rank (int32)
    input_dict = {
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "y": np.array([[3, 2, 1]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with same rank, different dimensions (int64)
    input_dict = {
        "x": np.array([[1], [2]], dtype=np.int64),
        "y": np.array([[2, 3, 4]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4-D numpy arrays (int8)
    input_dict = {
        "x": np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int8),
        "y": np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values (int64)
    input_dict = {
        "x": np.array([9223372036854775807, -9223372036854775808], dtype=np.int64),
        "y": np.array([0, 100], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar and array broadcasting (int32)
    input_dict = {
        "x": np.array([[1, -5], [10, 3]], dtype=np.int32),
        "y": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.max_2"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_2'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_2'], lib="jax", suffix=2)
