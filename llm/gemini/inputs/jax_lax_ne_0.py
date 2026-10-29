
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_ne_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays, identical shapes
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 0, 3, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 arrays with negative values
    x = np.array([[-1.5, 2.0], [0.0, -4.2]], dtype=np.float32)
    y = np.array([[-1.5, -2.0], [0.0, 4.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D (scalar) arrays, float64
    x = np.array(3.14159, dtype=np.float64)
    y = np.array(2.71828, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D broadcasting with same number of dimensions, int64
    x = np.array([[1], [-2], [3]], dtype=np.int64)
    y = np.array([[1, 2, 3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 arrays, broadcasting with matching number of dimensions
    x = np.ones((2, 1, 4), dtype=np.float32)
    y = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: boolean arrays, 1D
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int32 arrays, all identical
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array([[10, 20], [30, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D float64 arrays with zeros and negatives
    x = np.array([-0.0, 0.0, -5.5, 12.3], dtype=np.float64)
    y = np.array([0.0, -0.0, -5.5, -12.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D float32 arrays
    x = np.zeros((2, 2, 3, 1), dtype=np.float32)
    y = np.ones((2, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: float32 arrays containing NaN and Inf
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    y = np.array([np.nan, np.inf, np.inf, -0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D int64 arrays
    x = np.arange(12, dtype=np.int64).reshape((2, 2, 3))
    y = np.zeros((2, 2, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.ne"] = jax_lax_ne_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ne' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ne'.")


check_valid('jax.lax.ne', generated_inputs['jax.lax.ne'], lib="jax", suffix=0)
