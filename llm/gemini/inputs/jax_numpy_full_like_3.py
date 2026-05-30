
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def full_like_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "a": np.random.randn(3, 3).astype(np.float32),
        "fill_value": 5,
        "dtype": np.float32,
        "shape": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "a": np.random.randn(2, 4).astype(np.float64),
        "fill_value": -1,
        "dtype": np.int32,
        "shape": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "a": np.arange(10).astype(np.int32),
        "fill_value": 42,
        "dtype": np.int64,
        "shape": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "a": np.ones((5, 5), dtype=np.bool_),
        "fill_value": 0,
        "dtype": np.float64,
        "shape": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "a": np.random.randn(1).astype(np.float32),
        "fill_value": 100,
        "dtype": np.int32,
        "shape": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "a": np.zeros((2, 2, 2), dtype=np.int16),
        "fill_value": -99,
        "dtype": np.int16,
        "shape": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "a": np.arange(5, dtype=np.float32),
        "fill_value": 7,
        "dtype": np.float32,
        "shape": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "a": np.random.randn(4).astype(np.float64),
        "fill_value": -5,
        "dtype": np.int64,
        "shape": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "a": np.ones((3, 1), dtype=np.int32),
        "fill_value": 12,
        "dtype": np.float32,
        "shape": 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "a": np.zeros(10, dtype=np.float64),
        "fill_value": -123,
        "dtype": np.int32,
        "shape": 15
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.full_like_3"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_3'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_3'], lib="jax", suffix=3)
