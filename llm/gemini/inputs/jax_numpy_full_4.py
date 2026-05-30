
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def full_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "shape": 5,
        "fill_value": np.array(2.0, dtype=np.float32),
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "shape": 10,
        "fill_value": np.array(-1, dtype=np.int32),
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "shape": 3,
        "fill_value": np.array(True, dtype=np.bool_),
        "dtype": np.dtype(np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "shape": 1,
        "fill_value": np.array(3.14, dtype=np.float32),
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "shape": 8,
        "fill_value": np.array(100, dtype=np.int32),
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "shape": 12,
        "fill_value": np.array(False, dtype=np.bool_),
        "dtype": np.dtype(np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "shape": 4,
        "fill_value": np.array(-0.5, dtype=np.float32),
        "dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "shape": 6,
        "fill_value": np.array(0, dtype=np.int32),
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "shape": 15,
        "fill_value": np.array(True, dtype=np.bool_),
        "dtype": np.dtype(np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "shape": 2,
        "fill_value": np.array(12345, dtype=np.int32),
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.full_4"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_4'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_4'], lib="jax", suffix=4)
