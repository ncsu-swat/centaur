
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x = np.array([-1.5, 0.0, 2.3, -4.8], dtype=np.float32)
    y = np.array([0.5, -1.0, 2.0, 3.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 arrays with broadcasting (same rank)
    x = np.array([[-2.0], [3.5], [1.0]], dtype=np.float32)
    y = np.array([[1.0, -1.0, 2.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Scalar (0D tensor) and 2D array, float32
    x = np.array(0.0, dtype=np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D int32 arrays with negative values
    x = np.array([-10, 5, 0, -3], dtype=np.int32)
    y = np.array([2, -5, -1, -3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int64 arrays of same shape
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D boolean arrays
    x = np.array([[True, False], [True, True]], dtype=np.bool_)
    y = np.array([[False, False], [True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D float64 arrays with broadcasting (same rank)
    x = np.random.randn(2, 1, 4, 1).astype(np.float64)
    y = np.random.randn(1, 3, 1, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 0D scalars, int32
    x = np.array(-7, dtype=np.int32)
    y = np.array(12, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int32 arrays with broadcasting (same rank)
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[3, 2, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D float32 arrays
    x = np.random.randn(2, 3, 2).astype(np.float32)
    y = np.random.randn(2, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D float64 arrays
    x = np.array([np.nan, 1.0, -np.inf, np.inf], dtype=np.float64)
    y = np.array([2.0, np.nan, 0.0, -np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.min"] = lax_min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.min' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.min'.")


check_valid('jax.lax.min', generated_inputs['jax.lax.min'], lib="jax", suffix=0)
