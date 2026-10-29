
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def leaky_relu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like slope, float32
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=np.float32)
    negative_slope = 0.01
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 2: 2D array, standard LeakyReLU slope (0.2), float32
    x = np.random.randn(4, 5).astype(np.float32)
    negative_slope = 0.2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 3: 0D scalar array, float64
    x = np.array(-2.5, dtype=np.float64)
    negative_slope = 0.05
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 4: 3D array, negative slope is 0.0 (equivalent to standard ReLU), float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    negative_slope = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 5: 4D array (batch, channels, height, width), float32
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    negative_slope = 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 6: 2D array with float64 precision
    x = np.random.randn(6, 6).astype(np.float64)
    negative_slope = 0.33
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 7: 1D array with float16 precision
    x = np.array([-10.0, -0.5, 0.0, 0.5, 10.0], dtype=np.float16)
    negative_slope = 0.02
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 8: 5D array, small negative slope
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    negative_slope = 1e-4
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 9: 2D array with all negative values
    x = -np.abs(np.random.randn(5, 5)).astype(np.float32)
    negative_slope = 0.15
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 10: 2D array, slope > 1.0
    x = np.random.randn(3, 7).astype(np.float32)
    negative_slope = 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 11: 1D array with negative slope parameter as negative float
    x = np.random.randn(10).astype(np.float32)
    negative_slope = -0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

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
