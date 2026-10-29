
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def transpose_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 2: 2D int32 matrix transpose
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axes = [1, 0]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 3: 2D float64 identity transpose
    a = np.random.randn(4, 5).astype(np.float64)
    axes = [0, 1]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 4: 3D float32 swapping last two dimensions
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2, 1]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 5: 3D int64 cyclic permutation
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    axes = [2, 0, 1]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 6: 4D float32 array
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = [3, 2, 1, 0]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 7: 4D array with custom axes order
    a = np.random.randn(3, 4, 2, 5).astype(np.float32)
    axes = [0, 3, 1, 2]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 8: 5D float16 array
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float16)
    axes = [4, 1, 3, 0, 2]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 9: 2D boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    axes = [1, 0]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 10: 3D complex64 array
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex64
    )
    axes = [1, 2, 0]
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 11: 0D scalar array
    a = np.array(42.0, dtype=np.float32)
    axes = []
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 12: 6D int16 array reversed axes
    a = np.ones((2, 1, 3, 1, 4, 2), dtype=np.int16)
    axes = [5, 4, 3, 2, 1, 0]
    list_of_inputs.append({"a": a, "axes": axes})

    return list_of_inputs


generated_inputs["jax.numpy.transpose_2"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.transpose_2'.")


check_valid('jax.numpy.transpose', generated_inputs['jax.numpy.transpose_2'], lib="jax", suffix=2)
