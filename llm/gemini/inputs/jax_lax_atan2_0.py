
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atan2_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays (positive values)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D float32 arrays with negative and zero values
    x = np.array([-1.0, 0.0, -3.0], dtype=np.float32)
    y = np.array([2.0, -5.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D float32 arrays
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 3D float32 arrays
    x = np.random.randn(2, 2, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 1D float64 arrays
    x = np.random.randn(5).astype(np.float64)
    y = np.random.randn(5).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 2D float64 arrays
    x = np.random.randn(4, 2).astype(np.float64)
    y = np.random.randn(4, 2).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 1D float16 arrays
    x = np.random.randn(3).astype(np.float16)
    y = np.random.randn(3).astype(np.float16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 2D float16 arrays
    x = np.random.randn(3, 3).astype(np.float16)
    y = np.random.randn(3, 3).astype(np.float16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Broadcasting with same number of dimensions (2D)
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(1, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Broadcasting with same number of dimensions (2D) - float64
    x = np.random.randn(1, 1).astype(np.float64)
    y = np.random.randn(2, 2).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Large dimensions float32
    x = np.random.randn(100, 100).astype(np.float32)
    y = np.random.randn(100, 100).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 12: 3D Broadcasting with same number of dimensions
    x = np.random.randn(1, 2, 1).astype(np.float32)
    y = np.random.randn(2, 1, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.atan2"] = atan2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atan2'.")


check_valid('jax.lax.atan2', generated_inputs['jax.lax.atan2'], lib="jax", suffix=0)
