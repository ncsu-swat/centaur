
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def geomspace_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range, endpoint=True, float32, axis=0
    input_dict = {
        "start": 1,
        "stop": 1000,
        "num": 4,
        "endpoint": True,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Powers of two, endpoint=False, float64, axis=0
    input_dict = {
        "start": 2,
        "stop": 1024,
        "num": 10,
        "endpoint": False,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative range, endpoint=True, float32, axis=-1
    input_dict = {
        "start": -1,
        "stop": -100,
        "num": 3,
        "endpoint": True,
        "dtype": np.dtype("float32"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decreasing sequence, endpoint=True, float32, axis=0
    input_dict = {
        "start": 1000,
        "stop": 1,
        "num": 4,
        "endpoint": True,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Decreasing negative sequence, endpoint=False, float64, axis=-1
    input_dict = {
        "start": -1000,
        "stop": -1,
        "num": 5,
        "endpoint": False,
        "dtype": np.dtype("float64"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer dtype output, endpoint=True, int32, axis=0
    input_dict = {
        "start": 1,
        "stop": 16,
        "num": 5,
        "endpoint": True,
        "dtype": np.dtype("int32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element output, endpoint=True, float32, axis=0
    input_dict = {
        "start": 8,
        "stop": 64,
        "num": 1,
        "endpoint": True,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Base-3 geometric sequence, endpoint=True, float64, axis=-1
    input_dict = {
        "start": 3,
        "stop": 243,
        "num": 6,
        "endpoint": True,
        "dtype": np.dtype("float64"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative range with int64 dtype, endpoint=False, axis=0
    input_dict = {
        "start": -2,
        "stop": -32,
        "num": 5,
        "endpoint": False,
        "dtype": np.dtype("int64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Wide range, endpoint=False, float32, axis=-1
    input_dict = {
        "start": 10,
        "stop": 100000,
        "num": 5,
        "endpoint": False,
        "dtype": np.dtype("float32"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty sequence (num=0), endpoint=True, float32, axis=0
    input_dict = {
        "start": 1,
        "stop": 100,
        "num": 0,
        "endpoint": True,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.geomspace_3"] = geomspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.geomspace_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.geomspace_3'.")


check_valid('jax.numpy.geomspace', generated_inputs['jax.numpy.geomspace_3'], lib="jax", suffix=3)
