
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, linear detrend
    data = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    input_dict = {
        "data": data,
        "axis": -1,
        "type": "linear",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array with negative values, constant detrend
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "constant",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, detrend along axis 0
    data = np.random.randn(5, 10).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "linear",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, detrend along axis 1, constant
    data = np.random.randn(4, 8).astype(np.float64)
    input_dict = {
        "data": data,
        "axis": 1,
        "type": "constant",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, detrend along axis 2
    data = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 2,
        "type": "linear",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, detrend along negative axis
    data = np.random.randn(2, 5, 6).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": -1,
        "type": "constant",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with bp as a non-zero integer (breakpoint index)
    data = np.linspace(0, 10, 12, dtype=np.float32)
    input_dict = {
        "data": data,
        "axis": -1,
        "type": "linear",
        "bp": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, detrend along axis 1
    data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": 1,
        "type": "linear",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axis = -2, linear detrend
    data = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "data": data,
        "axis": -2,
        "type": "linear",
        "bp": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array, bp = 3, linear detrend
    data = np.array([1.5, 2.5, 4.0, 3.5, 5.0, 6.5], dtype=np.float64)
    input_dict = {
        "data": data,
        "axis": 0,
        "type": "linear",
        "bp": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.detrend_1"] = detrend_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.detrend_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.detrend_1'.")


check_valid('jax.scipy.signal.detrend', generated_inputs['jax.scipy.signal.detrend_1'], lib="jax", suffix=1)
