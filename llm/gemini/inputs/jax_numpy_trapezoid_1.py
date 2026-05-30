
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, integrating over default-like axis
    y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y1,
        "x": x1,
        "dx": 1.0,
        "axis": 0
    })

    # Input 2: 1D float64 arrays with negative values and irregular spacing
    y2 = np.array([-1.0, 2.5, -3.0, 4.2], dtype=np.float64)
    x2 = np.array([0.0, 1.5, 2.0, 4.5], dtype=np.float64)
    list_of_inputs.append({
        "y": y2,
        "x": x2,
        "dx": 0.5,
        "axis": -1
    })

    # Input 3: 2D array, integrating along axis 0
    y3 = np.random.randn(4, 3).astype(np.float32)
    x3 = np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y3,
        "x": x3,
        "dx": 2.0,
        "axis": 0
    })

    # Input 4: 2D array, integrating along axis 1
    y4 = np.random.randn(3, 5).astype(np.float64)
    x4 = np.array([0.1, 0.5, 1.2, 2.0, 3.5], dtype=np.float64)
    list_of_inputs.append({
        "y": y4,
        "x": x4,
        "dx": 1.5,
        "axis": 1
    })

    # Input 5: 3D array, integrating along axis 2
    y5 = np.random.randn(2, 3, 4).astype(np.float32)
    x5 = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append({
        "y": y5,
        "x": x5,
        "dx": 0.1,
        "axis": 2
    })

    # Input 6: 3D array, integrating along negative axis -2
    y6 = np.random.randn(2, 4, 3).astype(np.float32)
    x6 = np.array([1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y6,
        "x": x6,
        "dx": 1.0,
        "axis": -2
    })

    # Input 7: Large 1D array for high resolution integration
    x7 = np.linspace(0.0, np.pi, 100).astype(np.float64)
    y7 = np.sin(x7)
    list_of_inputs.append({
        "y": y7,
        "x": x7,
        "dx": 0.01,
        "axis": 0
    })

    # Input 8: 4D array, integrating along axis 1
    y8 = np.random.randn(2, 5, 2, 2).astype(np.float32)
    x8 = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y8,
        "x": x8,
        "dx": 2.0,
        "axis": 1
    })

    # Input 9: Decreasing x coordinates (reverse integration)
    y9 = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    x9 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y9,
        "x": x9,
        "dx": -1.0,
        "axis": 0
    })

    # Input 10: 2D array, integrating along axis -1 with multi-dimensional x matching y shape
    y10 = np.random.randn(3, 3).astype(np.float32)
    x10 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float32)
    list_of_inputs.append({
        "y": y10,
        "x": x10,
        "dx": 1.0,
        "axis": -1
    })

    return list_of_inputs

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
