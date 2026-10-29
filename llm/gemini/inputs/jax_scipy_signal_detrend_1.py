
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default linear detrend, axis 0, bp 0
    data = np.array([1.0, 4.0, 8.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, constant detrend, negative axis, bp 0
    data = np.linspace(-5.0, 15.0, 20, dtype=np.float64)
    input_dict = {"data": data, "axis": -1, "type": "constant", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, linear detrend with non-zero breakpoint
    data = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float32
    )
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, linear detrend along last axis
    data = np.random.randn(8, 16).astype(np.float32)
    input_dict = {"data": data, "axis": -1, "type": "linear", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, constant detrend along axis 0
    data = np.random.randn(12, 6).astype(np.float64)
    input_dict = {"data": data, "axis": 0, "type": "constant", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, linear detrend along axis 1 with breakpoint
    data = np.random.randn(5, 15).astype(np.float32)
    input_dict = {"data": data, "axis": 1, "type": "linear", "bp": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, linear detrend along axis 1
    data = np.random.randn(4, 10, 6).astype(np.float32)
    input_dict = {"data": data, "axis": 1, "type": "linear", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, constant detrend along axis -1
    data = np.random.randn(3, 4, 12).astype(np.float64)
    input_dict = {"data": data, "axis": -1, "type": "constant", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, linear detrend along axis 0 with breakpoint
    data = np.random.randn(14, 5, 4).astype(np.float32)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, linear detrend along axis 2
    data = np.random.randn(2, 3, 8, 4).astype(np.float32)
    input_dict = {"data": data, "axis": 2, "type": "linear", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, constant detrend along negative axis -2
    data = np.random.randn(3, 2, 9, 5).astype(np.float64)
    input_dict = {"data": data, "axis": -2, "type": "constant", "bp": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D complex array, linear detrend
    data = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex64)
    input_dict = {"data": data, "axis": 0, "type": "linear", "bp": 0}
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
