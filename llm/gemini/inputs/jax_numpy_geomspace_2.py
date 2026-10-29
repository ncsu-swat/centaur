
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def geomspace_inputs():
    list_of_inputs = []

    # Input 1: Basic positive range
    input_dict = {
        "start": 1.0,
        "stop": 1000.0,
        "num": 4,
        "endpoint": True,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: endpoint=False
    input_dict = {
        "start": 1.0,
        "stop": 16.0,
        "num": 4,
        "endpoint": False,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 dtype
    input_dict = {
        "start": 2.0,
        "stop": 1024.0,
        "num": 10,
        "endpoint": True,
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative range
    input_dict = {
        "start": -1000.0,
        "stop": -1.0,
        "num": 4,
        "endpoint": True,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Decreasing positive range
    input_dict = {
        "start": 100.0,
        "stop": 0.01,
        "num": 5,
        "endpoint": True,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element num=1
    input_dict = {
        "start": 5.0,
        "stop": 25.0,
        "num": 1,
        "endpoint": True,
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Decreasing negative range, endpoint=False
    input_dict = {
        "start": -1.0,
        "stop": -100.0,
        "num": 6,
        "endpoint": False,
        "dtype": np.dtype(np.float32),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: axis=-1
    input_dict = {
        "start": 0.5,
        "stop": 32.0,
        "num": 7,
        "endpoint": True,
        "dtype": np.dtype(np.float64),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small fractional numbers
    input_dict = {
        "start": 1e-4,
        "stop": 1e4,
        "num": 9,
        "endpoint": True,
        "dtype": np.dtype(np.float64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger num with complex dtype
    input_dict = {
        "start": 1.0,
        "stop": 100.0,
        "num": 20,
        "endpoint": False,
        "dtype": np.dtype(np.complex64),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.geomspace_2"] = geomspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.geomspace_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.geomspace_2'.")


check_valid('jax.numpy.geomspace', generated_inputs['jax.numpy.geomspace_2'], lib="jax", suffix=2)
