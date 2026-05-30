
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, positive values, axis=-1
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 arrays with negative values, axis=0
    y = np.array([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
    dx = np.array(0.5, dtype=np.float64)
    axis = 0
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 arrays, integrate along axis 1 (default axis is -1)
    y = np.random.randn(3, 5).astype(np.float32)
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    dx = np.array(2.0, dtype=np.float32)
    axis = -1
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 arrays, integrate along axis 0
    y = np.random.randn(4, 3).astype(np.float32)
    x = np.array([0.1, 0.5, 0.9, 1.3], dtype=np.float32)
    dx = np.array(0.4, dtype=np.float32)
    axis = 0
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 arrays, integrate along axis 2
    y = np.random.randn(2, 3, 4).astype(np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 2
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 arrays, integrate along axis 1
    y = np.random.randn(2, 4, 3).astype(np.float64)
    x = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    dx = np.array(2.0, dtype=np.float64)
    axis = 1
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D arrays, matching shape of x and y
    y = np.random.randn(3, 3).astype(np.float32)
    x = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0], [0.0, 2.0, 4.0]], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 1
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, integer dtype for y, float32 for x
    y = np.array([1, 4, 9, 16], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 arrays, integrate along axis 2
    y = np.random.randn(2, 2, 5, 2).astype(np.float32)
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    dx = np.array(0.2, dtype=np.float32)
    axis = 2
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32, large grid
    y = np.sin(np.linspace(0, 10, 100)).astype(np.float32)
    x = np.linspace(0, 10, 100).astype(np.float32)
    dx = np.array(0.1, dtype=np.float32)
    axis = 0
    input_dict = {"y": y, "x": x, "dx": dx, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trapezoid_3"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_3'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_3'], lib="jax", suffix=3)
