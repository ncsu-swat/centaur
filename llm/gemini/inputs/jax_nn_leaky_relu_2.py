
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def leaky_relu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 0D negative_slope float32
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
    negative_slope = np.array(0.01, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 2: 2D array, 0D negative_slope float64
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    negative_slope = np.array(0.1, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 3: 3D array, negative slope of 0.2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    negative_slope = np.array(0.2, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 4: 4D array, negative slope of 0.0 (equivalent to standard relu)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    negative_slope = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 5: Scalar 0D input x
    x = np.array(-5.0, dtype=np.float32)
    negative_slope = np.array(0.05, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 6: Float16 array
    x = np.array([[-1.5, 0.5], [2.0, -3.0]], dtype=np.float16)
    negative_slope = np.array(0.01, dtype=np.float16)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 7: Broadcastable 1D negative_slope matching trailing dimension
    x = np.random.randn(4, 5).astype(np.float32)
    negative_slope = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 8: Same shape array for negative_slope
    x = np.array([-1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    negative_slope = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 9: 5D array
    x = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
    negative_slope = np.array(0.01, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 10: Negative slope greater than 1.0
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float64)
    negative_slope = np.array(1.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

    # Input 11: Negative slope with negative value
    x = np.random.randn(3, 3).astype(np.float32)
    negative_slope = np.array(-0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "negative_slope": negative_slope})
    )

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
