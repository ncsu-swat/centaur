
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D array, linear detrend, axis -1, single breakpoint at 0
    data = np.linspace(0.0, 10.0, 20, dtype=np.float32) + np.random.randn(
        20
    ).astype(np.float32)
    axis = -1
    detrend_type = "linear"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 2: 1D array, constant detrend, axis 0, bp=(0,)
    data = np.array(
        [-5.0, -2.0, 0.0, 3.0, 7.0, 12.0, 15.0, 20.0], dtype=np.float64
    )
    axis = 0
    detrend_type = "constant"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 3: 2D array, linear detrend along axis 0, with internal breakpoint
    data = np.random.randn(30, 8).astype(np.float32)
    axis = 0
    detrend_type = "linear"
    bp = (0, 15)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 4: 2D array, constant detrend along axis 1
    data = np.random.randn(6, 12).astype(np.float64)
    axis = 1
    detrend_type = "constant"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 5: 3D array, linear detrend along axis 2 with multiple breakpoints
    data = np.random.randn(4, 5, 40).astype(np.float32)
    axis = 2
    detrend_type = "linear"
    bp = (0, 10, 25)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 6: 3D array, linear detrend along axis -1 with negative values
    data = (np.random.randn(3, 4, 16) - 5.0).astype(np.float32)
    axis = -1
    detrend_type = "linear"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 7: 3D array, constant detrend along middle axis (axis 1)
    data = np.random.randn(5, 18, 6).astype(np.float64)
    axis = 1
    detrend_type = "constant"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 8: 4D array, linear detrend along axis -1
    data = np.random.randn(2, 3, 4, 25).astype(np.float32)
    axis = -1
    detrend_type = "linear"
    bp = (0, 12)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 9: 2D array, negative axis index (axis=-2)
    data = np.random.randn(20, 10).astype(np.float64)
    axis = -2
    detrend_type = "linear"
    bp = (0, 8)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 10: 1D array, large size, linear detrend with multiple breakpoints
    data = np.sin(np.linspace(0, 50, 100)).astype(np.float32)
    axis = 0
    detrend_type = "linear"
    bp = (0, 25, 50, 75)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

    # Input 11: 4D array, constant detrend along axis 0
    data = np.random.randn(15, 2, 3, 4).astype(np.float32)
    axis = 0
    detrend_type = "constant"
    bp = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"data": data, "axis": axis, "type": detrend_type, "bp": bp}
        )
    )

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
