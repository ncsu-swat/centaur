
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, simple linear spacing
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": -1
    })

    # Input 2: 2D float64, integrating along axis 0
    y = np.random.randn(3, 4).astype(np.float64)
    x = np.array([0.0, 1.5, 3.0], dtype=np.float64)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": 0
    })

    # Input 3: 2D float32, integrating along axis 1
    y = np.random.randn(3, 4).astype(np.float32)
    x = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 2,
        "axis": 1
    })

    # Input 4: 3D float32, integrating along axis -1
    y = np.random.randn(2, 3, 5).astype(np.float32)
    x = np.linspace(0.0, 10.0, 5).astype(np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": -1
    })

    # Input 5: 3D float64, integrating along axis 1
    y = np.random.randn(2, 4, 3).astype(np.float64)
    x = np.linspace(-5.0, 5.0, 4).astype(np.float64)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": -2,
        "axis": 1
    })

    # Input 6: 1D with negative values and irregular spacing
    y = np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": 0
    })

    # Input 7: 2D where x has the same shape as y
    y = np.random.randn(3, 3).astype(np.float32)
    x = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": 1
    })

    # Input 8: 4D float32 tensor, axis = 2
    y = np.random.randn(2, 2, 3, 2).astype(np.float32)
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 3,
        "axis": 2
    })

    # Input 9: 1D int32 for y, float32 for x
    y = np.array([1, -2, 3, -4], dtype=np.int32)
    x = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": 0
    })

    # Input 10: 2D, negative axis = -2
    y = np.random.randn(4, 5).astype(np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "y": y,
        "x": x,
        "dx": 1,
        "axis": -2
    })

    return list_of_inputs

generated_inputs["jax.numpy.trapezoid_5"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_5'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_5'], lib="jax", suffix=5)
