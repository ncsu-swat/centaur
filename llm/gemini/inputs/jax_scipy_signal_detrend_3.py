
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, linear trend
    data = np.arange(10, dtype=np.float32) + np.random.normal(0, 0.1, 10).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": -1,
        "type": "linear",
        "bp": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, constant trend
    data = np.ones(15, dtype=np.float64) * 5.0 + np.random.normal(0, 0.5, 15).astype(np.float64)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "constant",
        "bp": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis=-1, linear trend
    data = np.random.randn(5, 10).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": -1,
        "type": "linear",
        "bp": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis=1, piecewise linear trend
    data = np.random.randn(8, 12).astype(np.float64)
    input_dict = {
        "data": data,
        "axis": 1,
        "type": "linear",
        "bp": (4,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, axis=0, constant trend
    data = np.random.randn(10, 4).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "constant",
        "bp": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis=2, piecewise linear trend
    data = np.random.randn(3, 4, 10).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 2,
        "type": "linear",
        "bp": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, axis=1, constant trend
    data = np.random.randn(5, 8, 3).astype(np.float64)
    input_dict = {
        "data": data,
        "axis": 1,
        "type": "constant",
        "bp": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array with negative values, piecewise linear
    data = np.linspace(-10, 10, 20, dtype=np.float32) + np.random.randn(20).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "linear",
        "bp": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axis=0, multiple breakpoints
    data = np.random.randn(15, 6).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "linear",
        "bp": (5, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, axis=1, multiple breakpoints
    data = np.random.randn(2, 12, 2).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 1,
        "type": "linear",
        "bp": (3, 7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.detrend_3"] = detrend_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.detrend_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.detrend_3'.")


check_valid('jax.scipy.signal.detrend', generated_inputs['jax.scipy.signal.detrend_3'], lib="jax", suffix=3)
