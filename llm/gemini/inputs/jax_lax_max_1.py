
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_jax_lax_max_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float32 arrays with negative values
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D int32 arrays
    x = np.array([-10, 0, 10], dtype=np.int32)
    y = np.array([-5, 5, -5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 2D int32 arrays with broadcasting (3, 1) and (1, 4)
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[4, 5, 6, 7]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: float64 arrays
    x = np.random.randn(5, 5).astype(np.float64)
    y = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: float16 arrays
    x = np.random.randn(3, 3).astype(np.float16)
    y = np.random.randn(3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 arrays
    x = np.array([-100, 200, -300], dtype=np.int64)
    y = np.array([100, -200, 300], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D float32 arrays
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 0D arrays (scalar-like)
    x = np.array(5.5, dtype=np.float32)
    y = np.array(4.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D int32 arrays with broadcasting (1, 3, 1) and (2, 1, 4)
    x = np.array([[[1], [2], [3]]], dtype=np.int32)
    y = np.array([[[4, 5, 6, 7]], [[8, 9, 10, 11]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 12: 1D float64 with large values
    x = np.array([1e10, -1e10, 0.0], dtype=np.float64)
    y = np.array([1e9, -1e9, 1e-9], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.lax.max_1"] = generate_jax_lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_1'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_1'], lib="jax", suffix=1)
