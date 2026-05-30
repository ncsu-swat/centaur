
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def leaky_relu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard float32, positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    negative_slope = 0.01
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 2: 2D array, float32
    x = np.random.randn(3, 3).astype(np.float32)
    negative_slope = 0.1
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 3: 3D array, float32
    x = np.random.randn(2, 4, 4).astype(np.float32)
    negative_slope = 0.2
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 4: 1D array, float64
    x = np.linspace(-5.0, 5.0, 10).astype(np.float64)
    negative_slope = 0.05
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 5: 4D array, float32
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)
    negative_slope = 0.01
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 6: 2D array with uniform distribution
    x = np.random.uniform(-10.0, 10.0, (5, 5)).astype(np.float32)
    negative_slope = 0.15
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 7: 1D array with single negative element
    x = np.array([-10.0], dtype=np.float32)
    negative_slope = 0.3
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 8: Large 2D array, float32
    x = np.random.randn(100, 100).astype(np.float32)
    negative_slope = 0.01
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 9: 3D array with zeros and extreme values
    x = np.array([[[-100.0, 0.0, 100.0]]], dtype=np.float32)
    negative_slope = 0.5
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 10: 2D array, float64, slope 0.0 (standard ReLU equivalent)
    x = np.random.randn(4, 4).astype(np.float64)
    negative_slope = 0.0
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    # Input 11: 5D array, float32
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    negative_slope = 0.02
    list_of_inputs.append({"x": copy.deepcopy(x), "negative_slope": negative_slope})

    return list_of_inputs

generated_inputs["jax.nn.leaky_relu_1"] = leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.leaky_relu_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.leaky_relu_1'.")


check_valid('jax.nn.leaky_relu', generated_inputs['jax.nn.leaky_relu_1'], lib="jax", suffix=1)
