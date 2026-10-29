
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def complex_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, same shape
    x = np.array([1.0, 2.0, -3.5, 4.0], dtype=np.float32)
    y = np.array([-0.5, 1.5, 2.0, -4.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D arrays, float64, same shape
    x = np.random.randn(3, 4).astype(np.float64)
    y = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar arrays, float32
    x = np.array(3.14, dtype=np.float32)
    y = np.array(-2.71, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D broadcast-compatible shapes (3, 1) and (1, 5), float32
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(1, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D arrays, float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D arrays, float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D arrays with broadcasting along one axis (4, 1) and (4, 6), float64
    x = np.random.randn(4, 1).astype(np.float64)
    y = np.random.randn(4, 6).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 0D scalar array and 2D array, float32
    x = np.array(1.0, dtype=np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D array and 0D scalar array, float64
    x = np.random.randn(5, 2).astype(np.float64)
    y = np.array(-0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D arrays with zeros, infinities, and NaNs, float32
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    y = np.array([1.0, -1.0, 0.0, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D broadcast-compatible shapes (1, 4, 1) and (2, 1, 3), float32
    x = np.random.randn(1, 4, 1).astype(np.float32)
    y = np.random.randn(2, 1, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.complex"] = complex_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.complex'.")


check_valid('jax.lax.complex', generated_inputs['jax.lax.complex'], lib="jax", suffix=0)
