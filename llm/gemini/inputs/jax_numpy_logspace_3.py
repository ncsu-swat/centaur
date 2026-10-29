
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logspace_inputs():
    list_of_inputs = []

    # Input 1: Basic standard usage
    input_dict = {
        "start": 0,
        "stop": 2,
        "num": 5,
        "endpoint": True,
        "base": 10,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: endpoint False, base 2, float64
    input_dict = {
        "start": 1,
        "stop": 4,
        "num": 10,
        "endpoint": False,
        "base": 2,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start and stop values
    input_dict = {
        "start": -3,
        "stop": -1,
        "num": 7,
        "endpoint": True,
        "base": 10,
        "dtype": np.dtype("float32"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decreasing sequence, start > stop
    input_dict = {
        "start": 5,
        "stop": 0,
        "num": 6,
        "endpoint": True,
        "base": 3,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: num=1 case
    input_dict = {
        "start": 2,
        "stop": 8,
        "num": 1,
        "endpoint": True,
        "base": 2,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: num=0 case
    input_dict = {
        "start": 0,
        "stop": 5,
        "num": 0,
        "endpoint": True,
        "base": 10,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex dtype
    input_dict = {
        "start": 1,
        "stop": 3,
        "num": 4,
        "endpoint": False,
        "base": 2,
        "dtype": np.dtype("complex64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Base = 1
    input_dict = {
        "start": 0,
        "stop": 10,
        "num": 15,
        "endpoint": True,
        "base": 1,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large num value
    input_dict = {
        "start": -2,
        "stop": 2,
        "num": 100,
        "endpoint": True,
        "base": 10,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: start == stop
    input_dict = {
        "start": 3,
        "stop": 3,
        "num": 8,
        "endpoint": False,
        "base": 5,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.logspace_3"] = logspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logspace_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logspace_3'.")


check_valid('jax.numpy.logspace', generated_inputs['jax.numpy.logspace_3'], lib="jax", suffix=3)
