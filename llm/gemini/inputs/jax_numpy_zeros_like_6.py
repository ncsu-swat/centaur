
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32 dtype string, 1D list shape
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = "float32"
    shape = [3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D array, int32 dtype string, different 2D shape
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    dtype = "int32"
    shape = [3, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 3D array with negative values, float32 dtype string, 3D list shape
    a = np.random.randn(2, 4, 3).astype(np.float32)
    dtype = "float32"
    shape = [2, 4, 3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: Boolean array, bool dtype string, 1D list shape
    a = np.array([True, False, True], dtype=np.bool_)
    dtype = "bool"
    shape = [5]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: Complex array, complex64 dtype string, 2D list shape
    a = np.array([[1.0 + 2.0j, -3.0 + 0.5j]], dtype=np.complex64)
    dtype = "complex64"
    shape = [2, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 4D array, uint8 dtype string, 4D list shape
    a = np.zeros((2, 1, 3, 2), dtype=np.uint8)
    dtype = "uint8"
    shape = [1, 2, 2, 3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: 0D (scalar) array, int32 dtype string, 0D list shape (empty list)
    a = np.array(42, dtype=np.int32)
    dtype = "int32"
    shape = []
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: Array with negative numbers, int16 dtype string, list shape with 1s
    a = np.array([-10, -20, -30], dtype=np.int16)
    dtype = "int16"
    shape = [1, 3, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: Array with zeros, float16 dtype string, empty dimension in shape
    a = np.zeros((0, 5), dtype=np.float16)
    dtype = "float16"
    shape = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: 2D array, int8 dtype string, larger 1D shape
    a = np.ones((4, 4), dtype=np.int8)
    dtype = "int8"
    shape = [8]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 11: 3D array, float32 dtype string, reshaped list
    a = np.ones((1, 2, 3), dtype=np.float32)
    dtype = "float32"
    shape = [3, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.zeros_like_6"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_6'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_6'], lib="jax", suffix=6)
