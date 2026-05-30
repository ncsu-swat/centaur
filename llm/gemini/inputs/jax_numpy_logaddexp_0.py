
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logaddexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: Simple 1D float64 arrays with negative values
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    x2 = np.array([-4.0, -5.0, -6.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 2D arrays of same shape
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Broadcasting (1, 3) and (3, 1)
    x1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x2 = np.array([[4.0], [5.0], [6.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 3D arrays of same shape
    x1 = np.random.randn(2, 2, 3).astype(np.float32)
    x2 = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Large values that would typically overflow exp(x)
    x1 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    x2 = np.array([1001.0, 1999.0, 3001.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Very small/negative values
    x1 = np.array([-1000.0, -2000.0, -3000.0], dtype=np.float64)
    x2 = np.array([-1001.0, -1999.0, -3001.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 0D scalar arrays
    x1 = np.array(5.0, dtype=np.float32)
    x2 = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: High dimensional (4D) arrays
    x1 = np.random.uniform(-10, 10, (2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(-10, 10, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Broadcasting with scalar-like 1D array
    x1 = np.random.randn(5, 5).astype(np.float32)
    x2 = np.array([1.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.logaddexp"] = logaddexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp'.")


check_valid('jax.numpy.logaddexp', generated_inputs['jax.numpy.logaddexp'], lib="jax", suffix=0)
