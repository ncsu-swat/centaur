
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logspace_inputs():
    list_of_inputs = []

    # Input 1: Basic base-10 logspace with float32
    input_dict = {
        "start": 0,
        "stop": 2,
        "num": 5,
        "endpoint": True,
        "base": 10,
        "dtype": np.float32,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Base-2 logspace excluding the endpoint
    input_dict = {
        "start": 1,
        "stop": 5,
        "num": 10,
        "endpoint": False,
        "base": 2,
        "dtype": np.float64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start to positive stop
    input_dict = {
        "start": -3,
        "stop": 3,
        "num": 7,
        "endpoint": True,
        "base": 10,
        "dtype": np.float32,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Base-3 logspace with float64
    input_dict = {
        "start": 0,
        "stop": 4,
        "num": 9,
        "endpoint": True,
        "base": 3,
        "dtype": np.float64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float16 dtype with negative bounds
    input_dict = {
        "start": -5,
        "stop": -1,
        "num": 5,
        "endpoint": False,
        "base": 2,
        "dtype": np.float16,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 dtype logspace
    input_dict = {
        "start": 2,
        "stop": 4,
        "num": 3,
        "endpoint": True,
        "base": 5,
        "dtype": np.complex64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element logspace
    input_dict = {
        "start": 1,
        "stop": 1,
        "num": 1,
        "endpoint": True,
        "base": 10,
        "dtype": np.float32,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Reverse direction logspace (start > stop)
    input_dict = {
        "start": 5,
        "stop": 1,
        "num": 5,
        "endpoint": False,
        "base": 10,
        "dtype": np.float64,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large number of points
    input_dict = {
        "start": 0,
        "stop": 10,
        "num": 100,
        "endpoint": True,
        "base": 2,
        "dtype": np.float32,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Base-10 with negative range and float64 dtype
    input_dict = {
        "start": -10,
        "stop": -2,
        "num": 10,
        "endpoint": False,
        "base": 10,
        "dtype": np.float64,
        "axis": 0
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
