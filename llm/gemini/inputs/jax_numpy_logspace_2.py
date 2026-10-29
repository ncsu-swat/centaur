
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logspace_inputs():
    list_of_inputs = []

    # Input 1: Basic standard usage
    input_dict = {
        "start": 0.0,
        "stop": 2.0,
        "num": 50,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Base 2, endpoint False, float64
    input_dict = {
        "start": 1.0,
        "stop": 3.0,
        "num": 10,
        "endpoint": False,
        "base": 2.0,
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start and axis -1
    input_dict = {
        "start": -2.0,
        "stop": 2.0,
        "num": 5,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Base e (natural logarithm base), num=1
    input_dict = {
        "start": 0.0,
        "stop": 1.0,
        "num": 1,
        "endpoint": True,
        "base": float(np.e),
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative range with endpoint False
    input_dict = {
        "start": -5.0,
        "stop": -1.0,
        "num": 20,
        "endpoint": False,
        "base": 10.0,
        "dtype": np.dtype(np.float64),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Decreasing range with complex64 dtype
    input_dict = {
        "start": 2.0,
        "stop": 0.0,
        "num": 15,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.complex64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional start and stop, base 4.0
    input_dict = {
        "start": 0.5,
        "stop": 1.5,
        "num": 8,
        "endpoint": False,
        "base": 4.0,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger num value
    input_dict = {
        "start": -1.0,
        "stop": 1.0,
        "num": 100,
        "endpoint": True,
        "base": 2.0,
        "dtype": np.dtype(np.float32),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Fractional base less than 1.0
    input_dict = {
        "start": 3.0,
        "stop": -3.0,
        "num": 7,
        "endpoint": True,
        "base": 0.5,
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty logspace (num=0)
    input_dict = {
        "start": 0.0,
        "stop": 0.0,
        "num": 0,
        "endpoint": False,
        "base": 10.0,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex128 dtype
    input_dict = {
        "start": 1.0,
        "stop": 5.0,
        "num": 25,
        "endpoint": True,
        "base": 10.0,
        "dtype": np.dtype(np.complex128),
        "axis": -1,
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
