
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logspace_inputs():
    list_of_inputs = []

    # Input 1: standard base 10 logspace with float32
    input_dict = {
        "start": 0.0,
        "stop": 2.0,
        "num": 5,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: endpoint=False
    input_dict = {
        "start": 0.0,
        "stop": 2.0,
        "num": 5,
        "endpoint": False,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: base 2.0
    input_dict = {
        "start": 0.0,
        "stop": 6.0,
        "num": 7,
        "endpoint": True,
        "base": 2.0,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 with negative bounds
    input_dict = {
        "start": -2.0,
        "stop": 2.0,
        "num": 10,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float64),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 type
    input_dict = {
        "start": 0.0,
        "stop": 3.0,
        "num": 4,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.complex64),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: axis = -1
    input_dict = {
        "start": 1.0,
        "stop": 5.0,
        "num": 5,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large number of elements
    input_dict = {
        "start": -5.0,
        "stop": 5.0,
        "num": 100,
        "endpoint": False,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: base e (natural logarithm space)
    input_dict = {
        "start": 0.0,
        "stop": 1.0,
        "num": 10,
        "endpoint": True,
        "base": float(np.e),
        "dtype": np.dtype(np.float64),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: fractional base 0.5
    input_dict = {
        "start": -2.0,
        "stop": 2.0,
        "num": 5,
        "endpoint": True,
        "base": 0.5,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: very close bounds
    input_dict = {
        "start": 1.0001,
        "stop": 1.0002,
        "num": 3,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.logspace_2"] = logspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logspace_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logspace_2'.")


check_valid('jax.numpy.logspace', generated_inputs['jax.numpy.logspace_2'], lib="jax", suffix=2)
