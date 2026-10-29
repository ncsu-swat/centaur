
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default linear detrending along axis -1
    data = np.array([1.0, 4.0, 8.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {"data": data, "axis": -1, "type": "linear", "bp": [0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, constant detrending along axis 0
    data = np.array([2.5, -1.0, 3.2, 0.0, 7.8, -4.5], dtype=np.float64)
    input_dict = {"data": data, "axis": 0, "type": "constant", "bp": [0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, linear detrending with breakpoints along axis 0
    data = np.random.randn(10, 4).astype(np.float32)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": [0, 5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, constant detrending along axis 1
    data = np.random.randn(6, 12).astype(np.float64)
    input_dict = {"data": data, "axis": 1, "type": "constant", "bp": [0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array with negative axis -2
    data = np.random.randn(8, 8).astype(np.float32)
    input_dict = {"data": data, "axis": -2, "type": "linear", "bp": [0, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, linear detrending along axis -1 with single breakpoint
    data = np.random.randn(3, 4, 15).astype(np.float32)
    input_dict = {"data": data, "axis": -1, "type": "linear", "bp": [6]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, constant detrending along axis 1
    data = np.random.randn(5, 10, 4).astype(np.float64)
    input_dict = {"data": data, "axis": 1, "type": "constant", "bp": [0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, linear detrending along axis 0
    data = np.random.randn(12, 3, 3).astype(np.float32)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": [0, 4, 8]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, linear detrending along axis 2
    data = np.random.randn(2, 3, 16, 2).astype(np.float32)
    input_dict = {"data": data, "axis": 2, "type": "linear", "bp": [0, 8]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with multiple breakpoints
    data = np.linspace(-5.0, 15.0, 20, dtype=np.float32)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": [0, 5, 12]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float64 array, linear detrending along axis -1
    data = (np.arange(30, dtype=np.float64) ** 1.5).reshape(5, 6)
    input_dict = {"data": data, "axis": -1, "type": "linear", "bp": [0, 3]}
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
