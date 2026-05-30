
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trapezoid_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, default axis
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with negative values and irregular spacing
    y = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    x = np.array([0.0, 1.5, 2.0, 3.5, 4.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 0.5, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D y, 1D x matching axis 1
    y = np.random.randn(3, 10).astype(np.float32)
    x = np.linspace(0.0, 1.0, 10).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": 0.1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D y, 2D x of same shape
    y = np.random.randn(4, 5).astype(np.float32)
    x = np.random.rand(4, 5).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D y, 1D x matching axis 1
    y = np.random.randn(2, 4, 3).astype(np.float32)
    x = np.array([0.1, 0.5, 0.9, 1.2], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 0.2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D y, 3D x of same shape
    y = np.random.randn(2, 3, 4).astype(np.float32)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.5, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 precision
    y = np.linspace(0, 10, 50).astype(np.float64)
    x = np.linspace(0, 10, 50).astype(np.float64)
    input_dict = {"y": y, "x": x, "dx": 0.2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays with integer-like floats, axis -1
    y = np.array([10., 20., 30.], dtype=np.float32)
    x = np.array([1., 2., 4.], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D y, 1D x matching axis 2
    y = np.random.randn(2, 2, 5, 2).astype(np.float32)
    x = np.linspace(0, 4, 5).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.0, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D y, 1D x matching axis 0
    y = np.random.randn(6, 3).astype(np.float32)
    x = np.linspace(-3, 3, 6).astype(np.float32)
    input_dict = {"y": y, "x": x, "dx": 1.2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trapezoid_4"] = trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_4'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_4'], lib="jax", suffix=4)
