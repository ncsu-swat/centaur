
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, default spacing, axis=-1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = 1.0
    axis = -1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 2: 2D array, 1D x along axis=0, float64
    y = np.linspace(0.0, 10.0, 20, dtype=np.float64).reshape(5, 4)
    x = np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype=np.float64)
    dx = 1.5
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 3: 2D arrays for both x and y, integrating along axis=1
    y = np.random.randn(4, 6).astype(np.float32)
    x = np.sort(np.random.rand(4, 6).astype(np.float32), axis=1)
    dx = 0.5
    axis = 1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 4: 3D array with negative axis=-2, 1D x
    y = np.random.randn(3, 7, 5).astype(np.float32)
    x = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    dx = 2.0 / 6.0
    axis = -2
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 5: 1D array with negative y values and negative dx
    y = np.array([-5.0, -3.0, -1.0, 0.0, 2.0, -1.0], dtype=np.float64)
    x = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.0], dtype=np.float64)
    dx = -2.0
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 6: 3D arrays for both x and y, axis=-1
    y = np.ones((2, 3, 4), dtype=np.float64)
    x = np.broadcast_to(np.arange(4, dtype=np.float64), (2, 3, 4)).copy()
    dx = 1.0
    axis = -1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 7: Integer y with float x, 2D array along axis=1
    y = np.random.randint(-10, 10, size=(6, 12), dtype=np.int32)
    x = np.linspace(0.0, 5.0, 12, dtype=np.float32)
    dx = 5.0 / 11.0
    axis = 1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 8: 4D array, 1D x along axis=1
    y = np.random.randn(2, 4, 3, 5).astype(np.float32)
    x = np.array([0.1, 0.5, 1.2, 2.0], dtype=np.float32)
    dx = 0.5
    axis = 1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 9: 1D non-uniformly spaced sample points x
    y = np.exp(-np.linspace(0, 5, 8)).astype(np.float64)
    x = np.array([0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 5.0], dtype=np.float64)
    dx = 0.1
    axis = -1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 10: Complex y, 1D float x, axis=0
    y = (
        np.random.randn(5, 5).astype(np.float32)
        + 1j * np.random.randn(5, 5).astype(np.float32)
    ).astype(np.complex64)
    x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    dx = 0.25
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 11: Minimal 2-element segment 1D array
    y = np.array([3.0, 7.0], dtype=np.float32)
    x = np.array([1.0, 2.0], dtype=np.float32)
    dx = 1.0
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.trapezoid_1"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_1'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_1'], lib="jax", suffix=1)
