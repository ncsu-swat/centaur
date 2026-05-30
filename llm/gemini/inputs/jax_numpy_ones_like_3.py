
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_ones_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, target shape [5], float32 dtype
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.float32
    shape = [5]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 2: 2D integer array with negative values, target shape [2, 3], int32 dtype
    a = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    dtype = np.int32
    shape = [2, 3]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 3: 3D bool array, target shape [1, 2, 2], bool dtype
    a = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    dtype = np.bool_
    shape = [1, 2, 2]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 4: 4D float64 array, target shape [2, 2, 2, 2], float64 dtype
    a = np.random.randn(2, 1, 2, 1).astype(np.float64)
    dtype = np.float64
    shape = [2, 2, 2, 2]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 5: 0D array (scalar), target shape [3], float32 dtype
    a = np.array(5.0, dtype=np.float32)
    dtype = np.float32
    shape = [3]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 6: 2D uint8 array, target shape [4, 4], uint8 dtype
    a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    dtype = np.uint8
    shape = [4, 4]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 7: 1D complex array, target shape [2], complex64 dtype
    a = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    dtype = np.complex64
    shape = [2]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 8: 3D int16 array, target shape [5, 1], int16 dtype
    a = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int16)
    dtype = np.int16
    shape = [5, 1]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 9: 2D float16 array, target shape [3, 2, 1], float16 dtype
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    dtype = np.float16
    shape = [3, 2, 1]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 10: 1D int64 array, target shape [10], int64 dtype
    a = np.array([1000000000, 2000000000], dtype=np.int64)
    dtype = np.int64
    shape = [10]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 11: 3D float array, target shape [2, 4], bool dtype
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    dtype = np.bool_
    shape = [2, 4]
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.ones_like_3"] = jax_numpy_ones_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_like_3'.")


check_valid('jax.numpy.ones_like', generated_inputs['jax.numpy.ones_like_3'], lib="jax", suffix=3)
