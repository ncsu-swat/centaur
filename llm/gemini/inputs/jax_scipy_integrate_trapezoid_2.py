
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, default spacing
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 2: 2D array, integrate along axis 0 with 1D x
    y = np.random.randn(5, 8).astype(np.float32)
    x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    dx = np.array(0.25, dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 3: 2D array, integrate along axis 1 with 1D x, float64
    y = np.random.randn(4, 12).astype(np.float64)
    x = np.sort(np.random.rand(12)).astype(np.float64)
    dx = np.array(0.1, dtype=np.float64)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 4: 2D arrays, matching shapes for x and y, negative axis
    y = np.random.randn(6, 7).astype(np.float32)
    x = np.repeat(np.linspace(0.0, 5.0, 7, dtype=np.float32)[None, :], 6, axis=0)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 5: 3D array, integrate along middle axis (axis 1)
    y = np.random.randn(3, 10, 4).astype(np.float32)
    x = np.linspace(-2.0, 2.0, 10, dtype=np.float32)
    dx = np.array(0.4, dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 6: 1D array with negative values
    y = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, -1.0], dtype=np.float32)
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 7: 2D array, matching shape float64 x, integrate along axis 0
    y = np.random.randn(8, 6).astype(np.float64)
    x = np.broadcast_to(np.linspace(0.0, 10.0, 8, dtype=np.float64)[:, None], (8, 6)).copy()
    dx = np.array(1.0, dtype=np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 8: 3D array with matching shape x, integrate along axis 2
    y = np.random.randn(2, 3, 5).astype(np.float32)
    x = np.tile(np.linspace(0.0, 2.0, 5, dtype=np.float32), (2, 3, 1))
    dx = np.array(0.5, dtype=np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 9: 4D array, integrate along axis 1 with 1D x
    y = np.random.randn(2, 6, 3, 4).astype(np.float32)
    x = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    dx = np.array(0.2, dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 10: 1D array, non-uniform grid spacing
    x = np.array([0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6], dtype=np.float32)
    y = np.sin(x)
    dx = np.array(0.5, dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    # Input 11: 2D array, negative axis=-2
    y = np.random.randn(7, 9).astype(np.float32)
    x = np.linspace(0.0, 3.0, 7, dtype=np.float32)
    dx = np.array(0.5, dtype=np.float32)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"y": y, "x": x, "dx": dx, "axis": axis}))

    return list_of_inputs

generated_inputs["jax.scipy.integrate.trapezoid_2"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.integrate.trapezoid_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.integrate.trapezoid_2'.")


check_valid('jax.scipy.integrate.trapezoid', generated_inputs['jax.scipy.integrate.trapezoid_2'], lib="jax", suffix=2)
