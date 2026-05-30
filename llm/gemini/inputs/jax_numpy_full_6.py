
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "shape": (2, 3),
        "fill_value": 3.14,
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "shape": (5,),
        "fill_value": -1.0,
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "shape": (4, 4, 4),
        "fill_value": 0.0,
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "shape": (1, 10),
        "fill_value": 99.9,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"shape": (), "fill_value": -5.5, "dtype": np.dtype("float32")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "shape": (2, 2, 2, 2),
        "fill_value": 2.718,
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "shape": (100,),
        "fill_value": -100.0,
        "dtype": np.dtype("int16"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "shape": (3, 1),
        "fill_value": 0.5,
        "dtype": np.dtype("float16"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "shape": (10, 10),
        "fill_value": 123.45,
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "shape": (2, 5, 2),
        "fill_value": -0.0,
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_6"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_6'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_6'], lib="jax", suffix=6)
