
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, scalar alpha (0-D array)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 2: 2D float32 array, scalar alpha (0-D array)
    x = np.random.randn(3, 4).astype(np.float32)
    alpha = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 3: 3D float32 array, 1D alpha with size 1
    x = np.random.randn(2, 3, 4).astype(np.float32)
    alpha = np.array([2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 4: float64 array, float64 alpha
    x = np.random.randn(5).astype(np.float64)
    alpha = np.array(0.5, dtype=np.float64)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 5: 4D float32 array with uniform distribution (including negative values)
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    alpha = np.array(0.8, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 6: Large positive and negative values
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    alpha = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 7: Scalar x and scalar alpha (0-D arrays)
    x = np.array(-1.5, dtype=np.float32)
    alpha = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 8: Same shape for x and alpha to test element-wise alpha
    x = np.random.randn(3, 3).astype(np.float32)
    alpha = np.random.uniform(0.1, 2.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 9: Broad-castable alpha (shape compatible with x)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    alpha = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)  # Broadcasts along the last dimension
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 10: High-dimensional 5D array
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    alpha = np.array(1.2, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    return list_of_inputs

generated_inputs["jax.nn.celu_1"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.celu_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.celu_1'.")


check_valid('jax.nn.celu', generated_inputs['jax.nn.celu_1'], lib="jax", suffix=1)
