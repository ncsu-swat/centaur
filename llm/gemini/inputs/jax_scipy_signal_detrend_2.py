
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D array, linear, single breakpoint
    input_dict = {
        "data": np.random.randn(10).astype(np.float32),
        "axis": -1,
        "type": "linear",
        "bp": [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, constant, no breakpoints
    input_dict = {
        "data": np.random.randn(20).astype(np.float32),
        "axis": 0,
        "type": "constant",
        "bp": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, linear, multiple breakpoints along axis 0
    input_dict = {
        "data": np.random.randn(15, 10).astype(np.float32),
        "axis": 0,
        "type": "linear",
        "bp": [5, 10]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, constant, axis 1, empty bp
    input_dict = {
        "data": np.random.randn(5, 30).astype(np.float64),
        "axis": 1,
        "type": "constant",
        "bp": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, linear, single breakpoint on axis 2
    input_dict = {
        "data": np.random.randn(4, 5, 8).astype(np.float32),
        "axis": 2,
        "type": "linear",
        "bp": [4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with negative values, linear, single breakpoint
    input_dict = {
        "data": np.random.uniform(-10.0, 10.0, size=(100,)).astype(np.float32),
        "axis": 0,
        "type": "linear",
        "bp": [50]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, constant, axis -1, empty bp
    input_dict = {
        "data": np.random.randn(8, 8).astype(np.float32),
        "axis": -1,
        "type": "constant",
        "bp": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, linear, multiple breakpoints
    input_dict = {
        "data": np.random.randn(12).astype(np.float32),
        "axis": 0,
        "type": "linear",
        "bp": [4, 8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, linear, axis -2, single breakpoint
    input_dict = {
        "data": np.random.randn(2, 3, 10, 5).astype(np.float32),
        "axis": -2,
        "type": "linear",
        "bp": [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, double precision, linear, multiple breakpoints
    input_dict = {
        "data": np.random.randn(50).astype(np.float64),
        "axis": -1,
        "type": "linear",
        "bp": [10, 20, 30, 40]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.detrend_2"] = detrend_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.detrend_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.detrend_2'.")


check_valid('jax.scipy.signal.detrend', generated_inputs['jax.scipy.signal.detrend_2'], lib="jax", suffix=2)
