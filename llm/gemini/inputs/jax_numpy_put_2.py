
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def put_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "a": np.zeros(5, dtype=np.int32),
        "ind": np.array([0, 2, 4], dtype=np.int32),
        "v": np.array([10, 20, 30], dtype=np.int32),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "a": np.zeros(5, dtype=np.int32),
        "ind": np.array([0, 2, 6], dtype=np.int32),
        "v": np.array([10, 20, 30], dtype=np.int32),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "a": np.zeros(5, dtype=np.int32),
        "ind": np.array([0, 2, 6], dtype=np.int32),
        "v": np.array([10, 20, 30], dtype=np.int32),
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "a": np.zeros((3, 5), dtype=np.float32),
        "ind": np.array([0, 7, 14], dtype=np.int32),
        "v": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "a": np.zeros((3, 5), dtype=np.float32),
        "ind": np.array([1, 2, 3, 4], dtype=np.int32),
        "v": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "a": np.zeros((2, 2, 2), dtype=np.float64),
        "ind": np.array([0, 3, 7], dtype=np.int32),
        "v": np.array([1.1, 2.2, 3.3], dtype=np.float64),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "a": np.ones(10, dtype=np.float32),
        "ind": np.array([1, 3, 5, 7, 9], dtype=np.int32),
        "v": np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "a": np.array([1, 2, 3, 4], dtype=np.int64),
        "ind": np.array([0, 1], dtype=np.int32),
        "v": np.array([9, 9], dtype=np.int64),
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "a": np.ones((4, 4), dtype=np.float32),
        "ind": np.array([15], dtype=np.int32),
        "v": np.array([99.9], dtype=np.float32),
        "mode": "clip",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "a": np.array([10, 20, 30, 40, 50, 60], dtype=np.int32),
        "ind": np.array([2, 4], dtype=np.int32),
        "v": np.array([0, 0], dtype=np.int32),
        "mode": "wrap",
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_2"] = put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_2'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_2'], lib="jax", suffix=2)
