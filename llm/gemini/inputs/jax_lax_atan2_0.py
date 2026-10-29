
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atan2_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array([2.0, 1.0, -3.0, -0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D arrays, float32
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar arrays, float64
    x = np.array(1.5, dtype=np.float64)
    y = np.array(-2.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D arrays, float64 with negative and positive values
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 2D arrays with broadcasting (same ndim), float16
    x = np.random.randn(4, 1).astype(np.float16)
    y = np.random.randn(4, 5).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D arrays, float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 1D complex64 arrays
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    y = np.array([2.0 - 1.0j, -1.0 - 2.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D complex128 arrays
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    y = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D broadcastable shapes, float32
    x = np.random.randn(2, 1, 4).astype(np.float32)
    y = np.random.randn(1, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D arrays with boundary values, float64
    x = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf], dtype=np.float64)
    y = np.array([1.0, -1.0, 0.0, -0.0, np.inf, -np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.atan2"] = atan2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atan2'.")


check_valid('jax.lax.atan2', generated_inputs['jax.lax.atan2'], lib="jax", suffix=0)
