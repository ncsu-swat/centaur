
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_trapezoid_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, integrating along axis 1 (columns)
    y = np.random.randn(3, 5).astype(np.float32)
    x = np.array([0.0, 2.0, 5.0, 7.0, 10.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, integrating along axis 0 (rows)
    y = np.random.randn(4, 3).astype(np.float32)
    x = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, integrating along axis 2
    y = np.random.randn(2, 3, 4).astype(np.float32)
    x = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 3, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same shape y and x
    y = np.random.randn(3, 4).astype(np.float32)
    x = np.tile(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), (3, 1))
    input_dict = {"y": y, "x": x, "dx": 1, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values and float64
    y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)
    x = np.array([-3.0, -2.0, -1.0, 0.0], dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": -1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 1D array
    y = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    x = np.array([0.0, 1.5, 3.0], dtype=np.float64)
    input_dict = {"y": y, "x": x, "dx": 4, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis 1, integers and float mix
    y = np.random.randn(3, 5, 2).astype(np.float32)
    x = np.arange(5, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 1, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis -1
    y = np.random.randn(2, 2, 2, 3).astype(np.float32)
    x = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 5, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis -2
    y = np.random.randn(5, 5).astype(np.float32)
    x = np.arange(5, dtype=np.float32)
    input_dict = {"y": y, "x": x, "dx": 2, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trapezoid_2"] = jax_numpy_trapezoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trapezoid_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trapezoid_2'.")


check_valid('jax.numpy.trapezoid', generated_inputs['jax.numpy.trapezoid_2'], lib="jax", suffix=2)
