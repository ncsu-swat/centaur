
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float arrays of the same size with negative and positive values
    x1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    x2 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D random float32 arrays of same size
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Broadcasting 1D array to 2D array
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Arrays containing special float values (nan, inf, -inf)
    x1 = np.array([[np.nan, 1.0], [2.0, np.inf]], dtype=np.float32)
    x2 = np.array([[3.0, np.nan], [-np.inf, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 2D integer arrays (int32)
    x1 = np.random.randint(-100, 100, size=(5, 2)).astype(np.int32)
    x2 = np.random.randint(-100, 100, size=(5, 2)).astype(np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 3D float64 arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Scalar-like array broadcasting to a 2D array
    x1 = np.array([10.0], dtype=np.float32)
    x2 = np.random.randn(4, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Broadcasting with multiple dimensions (1, 3, 1) vs (2, 3, 4)
    x1 = np.random.randn(1, 3, 1).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Large values and small float64 values
    x1 = np.array([1e10, -1e10, 1e-10], dtype=np.float64)
    x2 = np.array([1e9, -1e9, 2e-10], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 4D arrays with complex broadcasting shapes (1, 4, 1, 5) vs (2, 1, 3, 5)
    x1 = np.random.randn(1, 4, 1, 5).astype(np.float32)
    x2 = np.random.randn(2, 1, 3, 5).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmax_1"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_1'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_1'], lib="jax", suffix=1)
