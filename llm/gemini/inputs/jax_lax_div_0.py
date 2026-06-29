
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def div_inputs():
    list_of_inputs = []

    # Input 1: 2D Float32 arrays
    x = np.random.randn(3, 3).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32) + 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D Int32 arrays
    x = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    y = np.random.randint(1, 10, size=(4, 4)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 1D Float64 arrays with negative values
    x = np.array([-1.5, -2.5, 3.5, 4.5], dtype=np.float64)
    y = np.array([0.5, -1.0, 2.0, -0.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 0D arrays (scalars as tensors) of Float32
    x = np.array(5.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcastable arrays of shape (3, 1) and (3, 3) of Float32
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32) + 2.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 3D Float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32) + 1.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 2D Int32 arrays of size (2, 2)
    x = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    y = np.random.randint(1, 5, size=(2, 2)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Broadcastable arrays of shape (1, 5) and (2, 5) of Float64
    x = np.random.randn(1, 5).astype(np.float64)
    y = np.random.randn(2, 5).astype(np.float64) - 2.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 4D Float32 arrays
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2).astype(np.float32) + 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 1D Int32 arrays
    x = np.random.randint(1, 100, size=(10,)).astype(np.int32)
    y = np.random.randint(1, 10, size=(10,)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Large 1D Float32 arrays
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32) + 5.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.div"] = div_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.div'.")


check_valid('jax.lax.div', generated_inputs['jax.lax.div'], lib="jax", suffix=0)
