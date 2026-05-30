
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argwhere_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "a": np.array([1, 0, 2], dtype=np.int32),
        "size": 2,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "a": np.array([[1, 0], [0, 2]], dtype=np.int32),
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "a": np.array([[1, 1], [0, 1]], dtype=np.int32),
        "size": 3,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "a": np.array([True, False, True], dtype=bool),
        "size": 2,
        "fill_value": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "a": np.array([[-1, 0], [0, -3]], dtype=np.int32),
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "a": np.array([[[0, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.int32),
        "size": 1,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "a": np.array(5, dtype=np.int32),
        "size": 1,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "a": np.array([[0.5, 0.0, 1.2], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]], dtype=np.float32),
        "size": 3,
        "fill_value": -9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "a": np.array([0, 0], dtype=np.int32),
        "size": 0,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "a": np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [3.1, 0.0]]], dtype=np.float64),
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argwhere_2"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_2'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_2'], lib="jax", suffix=2)
