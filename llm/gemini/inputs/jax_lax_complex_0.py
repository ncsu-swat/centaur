
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def complex_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 1D float32 arrays with negative and fractional values
    x = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    y = np.array([3.5, -4.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 2D float64 arrays
    x = np.random.randn(3, 3).astype(np.float64)
    y = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 3D float32 arrays
    x = np.random.randn(2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 0D arrays (scalars represented as tensors)
    x = np.array(1.5, dtype=np.float32)
    y = np.array(-2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Broadcasting (same number of dimensions, broadcast-compatible)
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(1, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Broadcasting in 3D
    x = np.random.randn(2, 1, 3).astype(np.float32)
    y = np.random.randn(1, 4, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: High-dimensional 4D float64 arrays
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    y = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Large 1D arrays
    x = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    y = np.linspace(10.0, -10.0, 100).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 5D float32 arrays
    x = np.ones((2, 1, 2, 1, 2), dtype=np.float32)
    y = np.zeros((2, 1, 2, 1, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

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
