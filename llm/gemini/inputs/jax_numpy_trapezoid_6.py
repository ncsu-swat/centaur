
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array integrating over axis 1
    y = np.random.randn(3, 5).astype(np.float32)
    x = np.sort(np.random.rand(5).astype(np.float32))
    dx = np.array(0.5, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array integrating over axis 0 with float64
    y = np.random.randn(4, 3).astype(np.float64)
    x = np.sort(np.random.rand(4).astype(np.float64))
    dx = np.array(0.1, dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, negative axis
    y = np.random.randn(2, 3, 4).astype(np.float32)
    x = np.linspace(0.0, 10.0, 4).astype(np.float32)
    dx = np.array(2.5, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values
    y = np.array([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float32)
    dx = np.array(0.2, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, multidimensional x matching y shape
    y = np.random.randn(3, 3).astype(np.float32)
    x = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0], [0.0, 0.5, 1.0]], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, middle axis
    y = np.random.randn(3, 5, 2).astype(np.float32)
    x = np.linspace(-1.0, 1.0, 5).astype(np.float32)
    dx = np.array(0.4, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, large size, float64
    y = np.sin(np.linspace(0, np.pi, 100)).astype(np.float64)
    x = np.linspace(0, np.pi, 100).astype(np.float64)
    dx = np.array(0.01, dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis 2
    y = np.random.randn(2, 2, 5, 2).astype(np.float32)
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    dx = np.array(1.0, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis -2
    y = np.random.randn(4, 2).astype(np.float32)
    x = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
    dx = np.array(0.1, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": dx, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trapezoid_6"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_6'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_6'], lib="jax", suffix=6)
