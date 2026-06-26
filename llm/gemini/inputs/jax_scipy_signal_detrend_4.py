
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default linear detrend, single breakpoint at 0
    data = np.random.randn(15).astype(np.float32)
    axis = -1
    type_str = 'linear'
    bp = np.array([0], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, constant detrend, single breakpoint at 0
    data = np.random.randn(20).astype(np.float64)
    axis = 0
    type_str = 'constant'
    bp = np.array([0], dtype=np.int64)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, linear detrend along last axis, breakpoint at 4
    data = np.random.randn(5, 10).astype(np.float32)
    axis = 1
    type_str = 'linear'
    bp = np.array([4], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, linear detrend along axis 0, multiple breakpoints
    data = np.random.randn(12, 6).astype(np.float32)
    axis = 0
    type_str = 'linear'
    bp = np.array([3, 7], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array, constant detrend along axis 1, breakpoint at 0
    data = np.random.randn(3, 8, 4).astype(np.float64)
    axis = 1
    type_str = 'constant'
    bp = np.array([0], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32 array with linear trend, breakpoint at 10
    data = np.linspace(-10, 10, 20).astype(np.float32)
    axis = 0
    type_str = 'linear'
    bp = np.array([10], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array, constant detrend, multiple breakpoints along last axis
    data = np.random.randn(4, 15).astype(np.float64)
    axis = -1
    type_str = 'constant'
    bp = np.array([5, 10], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, linear detrend along axis 1, multiple breakpoints
    data = np.random.randn(2, 8, 3).astype(np.float32)
    axis = 1
    type_str = 'linear'
    bp = np.array([2, 5], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D float32 array, linear detrend, multiple breakpoints
    data = np.random.randn(50).astype(np.float32)
    axis = 0
    type_str = 'linear'
    bp = np.array([15, 30], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D square float64 array, constant detrend, breakpoint at 0
    data = np.random.randn(10, 10).astype(np.float64)
    axis = 0
    type_str = 'constant'
    bp = np.array([0], dtype=np.int32)
    input_dict = {"data": data, "axis": axis, "type": type_str, "bp": bp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.detrend_4"] = detrend_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.detrend_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.detrend_4'.")


check_valid('jax.scipy.signal.detrend', generated_inputs['jax.scipy.signal.detrend_4'], lib="jax", suffix=4)
