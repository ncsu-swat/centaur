
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def collapse_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D array, collapsing first two dimensions
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Collapsing last two dimensions of a 3D array
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 1,
        "stop_dimension": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D integer array, collapsing middle two dimensions
    operand = np.random.randint(0, 10, size=(2, 5, 5, 3)).astype(np.int32)
    input_dict = {
        "operand": operand,
        "start_dimension": 1,
        "stop_dimension": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Collapsing all dimensions of a 4D float64 array
    operand = np.random.randn(3, 3, 3, 3).astype(np.float64)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High-dimensional array, collapsing a subset of dimensions
    operand = np.random.randn(2, 3, 2, 4, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 1,
        "stop_dimension": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, collapsing first dimension (no-op collapse)
    operand = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex numbers array
    operand = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean array, collapsing all dimensions
    operand = (np.random.randn(2, 3, 2) > 0).astype(np.bool_)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions
    operand = np.random.randn(10, 10, 10).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array containing a dimension of size 1
    operand = np.random.randn(5, 1, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_dimension": 0,
        "stop_dimension": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.collapse"] = collapse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.collapse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.collapse'.")


check_valid('jax.lax.collapse', generated_inputs['jax.lax.collapse'], lib="jax", suffix=0)
