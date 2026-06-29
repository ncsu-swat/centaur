
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_add_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, identical shapes
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D arrays, int32, identical shapes, with negative values
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D arrays, float64, identical shapes
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D arrays, complex64, identical shapes
    x = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    y = np.array([5 + 6j, 7 + 8j], dtype=np.complex64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D arrays, broadcast compatible (same dims: 2), float32
    x = np.random.randn(3, 1).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 2D arrays, broadcast compatible (same dims: 2), int16
    x = np.array([[10, 20, 30]], dtype=np.int16)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 4D arrays, identical shapes, float32
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 0D arrays (representing scalars as tensors), float32
    x = np.array(3.14, dtype=np.float32)
    y = np.array(2.71, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 3D arrays, broadcast compatible (same dims: 3), uint8
    x = np.ones((2, 1, 3), dtype=np.uint8)
    y = np.ones((2, 4, 3), dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D arrays, int64, identical shapes
    x = np.array([1000000000, 2000000000], dtype=np.int64)
    y = np.array([3000000000, 4000000000], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: 2D arrays, float16, identical shapes
    x = np.random.randn(5, 5).astype(np.float16)
    y = np.random.randn(5, 5).astype(np.float16)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.add"] = jax_lax_add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.add'.")


check_valid('jax.lax.add', generated_inputs['jax.lax.add'], lib="jax", suffix=0)
