
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def kron_inputs():
    list_of_inputs = []

    # Input 1: 2D integer matrices
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 2: 1D arrays
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([-4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 3: Different dimensions (1D and 2D)
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([[3, 4], [5, 6]], dtype=np.int64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 4: Different dimensions (2D and 1D)
    a = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 5: 3D arrays
    a = np.random.randn(2, 2, 2).astype(np.float32)
    b = np.random.randn(3, 1, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 6: 0D arrays (scalars as 0-dim arrays)
    a = np.array(3.5, dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 7: Complex numbers
    a = np.array([[1 + 2j, -1j], [3 - 1j, 4j]], dtype=np.complex64)
    b = np.array([[2j, 1], [-1, 2 - 3j]], dtype=np.complex64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 8: Unsigned integer arrays
    a = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    b = np.array([[3, 4], [5, 6]], dtype=np.uint8)
    list_of_inputs.append({"a": a, "b": b})

    # Input 9: Arrays with zero dimension (empty)
    a = np.empty((0, 2), dtype=np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 10: Higher-dimensional arrays (4D and 2D)
    a = np.random.randn(2, 1, 3, 2).astype(np.float32)
    b = np.random.randn(2, 4).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 11: 4D arrays with matching dimensions
    a = np.random.randn(1, 2, 1, 3).astype(np.float64)
    b = np.random.randn(2, 1, 2, 1).astype(np.float64)
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs


generated_inputs["jax.numpy.kron"] = kron_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kron'.")


check_valid('jax.numpy.kron', generated_inputs['jax.numpy.kron'], lib="jax", suffix=0)
