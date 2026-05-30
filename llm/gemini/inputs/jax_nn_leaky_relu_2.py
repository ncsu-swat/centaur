
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def leaky_relu_inputs():
    list_of_inputs = []

    # Input 1: 1D array float32, scalar-like negative_slope
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    negative_slope = np.array(0.01, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 2: 2D array float32, scalar-like negative_slope
    x = np.random.randn(3, 4).astype(np.float32)
    negative_slope = np.array(0.1, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 3: 3D array float64, scalar-like negative_slope
    x = np.random.randn(2, 3, 4).astype(np.float64)
    negative_slope = np.array(0.05, dtype=np.float64)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 4: 4D array float32, scalar-like negative_slope
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    negative_slope = np.array(0.2, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 5: 1D array float64, scalar-like negative_slope
    x = np.linspace(-10, 10, 10).astype(np.float64)
    negative_slope = np.array(0.01, dtype=np.float64)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 6: 2D array float16, scalar-like negative_slope
    x = np.random.randn(5, 5).astype(np.float16)
    negative_slope = np.array(0.02, dtype=np.float16)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 7: 1D array with 1D negative_slope (shape (1,))
    x = np.random.randn(10).astype(np.float32)
    negative_slope = np.array([0.15], dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 8: 3D array with negative slope as a negative value
    x = np.random.randn(2, 2, 2).astype(np.float32)
    negative_slope = np.array(-0.01, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 9: Large 2D array float32, scalar-like negative_slope
    x = np.random.randn(100, 100).astype(np.float32)
    negative_slope = np.array(0.01, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    # Input 10: 5D array float32, scalar-like negative_slope
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    negative_slope = np.array(0.1, dtype=np.float32)
    list_of_inputs.append({"x": x, "negative_slope": negative_slope})

    return list_of_inputs

generated_inputs["jax.nn.leaky_relu_2"] = leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.leaky_relu_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.leaky_relu_2'.")


check_valid('jax.nn.leaky_relu', generated_inputs['jax.nn.leaky_relu_2'], lib="jax", suffix=2)
