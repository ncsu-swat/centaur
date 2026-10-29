
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def detrend_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default linear detrending, bp=np.array([0])
    data = np.array([1.0, 4.0, 8.0, 8.0, 9.0], dtype=np.float32)
    axis = -1
    detrend_type = "linear"
    bp = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 2: 1D float64 array, constant detrending, scalar array bp
    data = np.array([2.5, -1.0, 3.2, 7.8, -4.5, 0.0], dtype=np.float64)
    axis = 0
    detrend_type = "constant"
    bp = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 3: 2D float32 array, detrending along axis 0
    data = np.random.randn(10, 4).astype(np.float32)
    axis = 0
    detrend_type = "linear"
    bp = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 4: 2D float32 array, constant detrending along axis 1
    data = np.random.randn(6, 8).astype(np.float32)
    axis = 1
    detrend_type = "constant"
    bp = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 5: 2D float64 array, linear detrending with a breakpoint along axis -1
    data = np.linspace(-5.0, 15.0, 20, dtype=np.float64).reshape(2, 10)
    axis = -1
    detrend_type = "linear"
    bp = np.array([5], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 6: 3D float32 array, linear detrending with multiple breakpoints along axis -1
    data = np.random.randn(3, 4, 12).astype(np.float32)
    axis = -1
    detrend_type = "linear"
    bp = np.array([4, 8], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 7: 3D float64 array, constant detrending along axis 1
    data = np.random.randn(2, 5, 3).astype(np.float64)
    axis = 1
    detrend_type = "constant"
    bp = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 8: 1D float32 array with negative linear trend and single breakpoint
    data = np.array(
        [-10.0, -8.0, -6.0, -2.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32
    )
    axis = -1
    detrend_type = "linear"
    bp = np.array([3], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 9: 4D float32 array, linear detrending along axis 2
    data = np.random.randn(2, 2, 8, 2).astype(np.float32)
    axis = 2
    detrend_type = "linear"
    bp = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 10: 2D float64 array with negative axis indexing, constant detrending
    data = np.random.randn(7, 5).astype(np.float64)
    axis = -2
    detrend_type = "constant"
    bp = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

    # Input 11: 1D float64 array, linear detrending with multiple internal breakpoints
    data = np.linspace(0.0, 100.0, 15, dtype=np.float64)
    axis = 0
    detrend_type = "linear"
    bp = np.array([3, 7, 11], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"data": data, "axis": axis, "type": detrend_type, "bp": bp})
    )

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
