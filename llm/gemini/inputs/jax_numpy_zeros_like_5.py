
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, float32 dtype string, shape 4
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = "float32"
    shape = 4
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D int32 array, int64 dtype string, shape 10
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    dtype = "int64"
    shape = 10
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 3D bool array, bool dtype string, shape 1
    a = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    dtype = "bool"
    shape = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: 0D float64 array, float64 dtype string, shape 0 (empty 1D)
    a = np.array(42.0, dtype=np.float64)
    dtype = "float64"
    shape = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: 2D uint8 array, int8 dtype string, shape 8
    a = np.zeros((3, 3), dtype=np.uint8)
    dtype = "int8"
    shape = 8
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 4D float32 array, complex64 dtype string, shape 3
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    dtype = "complex64"
    shape = 3
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: 1D complex128 array, float16 dtype string, shape 7
    a = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    dtype = "float16"
    shape = 7
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: 2D int16 array, uint32 dtype string, shape 16
    a = np.array([[-10, 20], [30, -40]], dtype=np.int16)
    dtype = "uint32"
    shape = 16
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: 3D int64 array with negative numbers, int32 dtype string, shape 2
    a = np.array([[[-5, -1], [2, 3]]], dtype=np.int64)
    dtype = "int32"
    shape = 2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: 1D empty array, float32 dtype string, shape 5
    a = np.empty((0,), dtype=np.float32)
    dtype = "float32"
    shape = 5
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 11: 5D float64 array, complex128 dtype string, shape 12
    a = np.ones((1, 2, 1, 3, 1), dtype=np.float64)
    dtype = "complex128"
    shape = 12
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.zeros_like_5"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_5'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_5'], lib="jax", suffix=5)
