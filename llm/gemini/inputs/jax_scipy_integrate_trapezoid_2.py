
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, simple linear spacing
    y = np.random.randn(10).astype(np.float32)
    x = np.linspace(0, 1, 10).astype(np.float32)
    dx = np.array(0.1, dtype=np.float32)
    axis = -1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 2: 2D array, float64, integrate along axis 0
    y = np.random.randn(5, 8).astype(np.float64)
    x = np.linspace(0, 2, 5).astype(np.float64)
    dx = np.array(0.5, dtype=np.float64)
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 3: 2D array, float32, integrate along axis 1
    y = np.random.randn(4, 12).astype(np.float32)
    x = np.linspace(0, 11, 12).astype(np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 4: 3D array, float32, integrate along axis 2
    y = np.random.randn(3, 4, 5).astype(np.float32)
    x = np.linspace(0, 1, 5).astype(np.float32)
    dx = np.array(0.2, dtype=np.float32)
    axis = 2
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 5: 3D array, negative axis integration
    y = np.random.randn(2, 3, 4).astype(np.float32)
    x = np.linspace(0, 1, 3).astype(np.float32)
    dx = np.array(0.5, dtype=np.float32)
    axis = -2
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 6: 1D array with irregular grid
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 7: 4D array, float64, axis 1
    y = np.random.randn(2, 3, 2, 5).astype(np.float64)
    x = np.linspace(0, 10, 3).astype(np.float64)
    dx = np.array(5.0, dtype=np.float64)
    axis = 1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 8: 1D array, strictly negative values
    y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 9: 2D array, simple integer-valued floats, axis -1
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    axis = -1
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

    # Input 10: 1D array representing a sine wave
    x = np.linspace(0, np.pi, 100).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    dx = np.array(np.pi / 100.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append({"y": y, "x": x, "dx": dx, "axis": axis})

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
