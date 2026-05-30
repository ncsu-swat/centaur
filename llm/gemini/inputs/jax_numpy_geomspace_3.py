
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def geomspace_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "start": 1,
        "stop": 1000,
        "num": 4,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 2
    list_of_inputs.append({
        "start": 2,
        "stop": 16,
        "num": 4,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 3
    list_of_inputs.append({
        "start": 10,
        "stop": 100000,
        "num": 5,
        "endpoint": True,
        "dtype": np.float32,
        "axis": -1
    })

    # Input 4
    list_of_inputs.append({
        "start": 1,
        "stop": 256,
        "num": 9,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 5
    list_of_inputs.append({
        "start": 3,
        "stop": 243,
        "num": 5,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 6
    list_of_inputs.append({
        "start": 5,
        "stop": 125,
        "num": 3,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 7
    list_of_inputs.append({
        "start": -1,
        "stop": -1000,
        "num": 4,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 8
    list_of_inputs.append({
        "start": -2,
        "stop": -32,
        "num": 5,
        "endpoint": False,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 9
    list_of_inputs.append({
        "start": 1,
        "stop": 1024,
        "num": 11,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

    # Input 10
    list_of_inputs.append({
        "start": 8,
        "stop": 64,
        "num": 2,
        "endpoint": True,
        "dtype": np.float32,
        "axis": 0
    })

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
