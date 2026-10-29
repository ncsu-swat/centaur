
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def linspace_inputs():
    list_of_inputs = []

    # Input 1: Basic range with default-like settings
    input_dict = {
        "start": 0.0,
        "stop": 10.0,
        "num": 50,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative to positive, endpoint False, float64
    input_dict = {
        "start": -5.5,
        "stop": 5.5,
        "num": 11,
        "endpoint": False,
        "retstep": False,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With retstep=True
    input_dict = {
        "start": 1.0,
        "stop": 100.0,
        "num": 10,
        "endpoint": True,
        "retstep": True,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single element, axis=-1
    input_dict = {
        "start": 0.0,
        "stop": 1.0,
        "num": 1,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype("float64"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Decreasing order, retstep=True, endpoint=False, axis=-1
    input_dict = {
        "start": 10.0,
        "stop": 0.0,
        "num": 5,
        "endpoint": False,
        "retstep": True,
        "dtype": np.dtype("float32"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex dtype
    input_dict = {
        "start": -100.0,
        "stop": -10.0,
        "num": 20,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype("complex64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty output with num=0
    input_dict = {
        "start": 0.0,
        "stop": 5.0,
        "num": 0,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype("float32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High precision values with float64
    input_dict = {
        "start": 3.141592653589793,
        "stop": 6.283185307179586,
        "num": 100,
        "endpoint": False,
        "retstep": False,
        "dtype": np.dtype("float64"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer output dtype
    input_dict = {
        "start": 0.0,
        "stop": 20.0,
        "num": 11,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype("int32"),
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scientific notation range with float16
    input_dict = {
        "start": 1e-3,
        "stop": 1e3,
        "num": 25,
        "endpoint": False,
        "retstep": True,
        "dtype": np.dtype("float16"),
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs


generated_inputs["jax.numpy.linspace_1"] = linspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linspace_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linspace_1'.")


check_valid('jax.numpy.linspace', generated_inputs['jax.numpy.linspace_1'], lib="jax", suffix=1)
