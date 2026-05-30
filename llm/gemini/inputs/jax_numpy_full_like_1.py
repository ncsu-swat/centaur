
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def full_like_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, 1D fill_value, float32
    a = np.arange(5, dtype=np.float32)
    fill_value = np.array([2.5], dtype=np.float32)
    dtype = np.dtype('float32')
    shape = (5,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 2: 2D array, broadcasting 2D fill_value, int32
    a = np.ones((2, 2), dtype=np.int32)
    fill_value = np.array([[9]], dtype=np.int32)
    dtype = np.dtype('int32')
    shape = (2, 3)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 3: 3D array, scalar-like array fill_value, float32
    a = np.random.randn(3, 4, 5).astype(np.float32)
    fill_value = np.array(-1.0, dtype=np.float32)
    dtype = np.dtype('float32')
    shape = (3, 4, 5)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 4: Boolean arrays
    a = np.array([True, False, True], dtype=bool)
    fill_value = np.array([True], dtype=bool)
    dtype = np.dtype('bool')
    shape = (4,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 5: Negative-to-positive integer array, int32
    a = np.arange(-10, 10).reshape(4, 5).astype(np.int32)
    fill_value = np.array([-5, -4, -3, -2, -1], dtype=np.int32)
    dtype = np.dtype('int32')
    shape = (2, 5)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 6: 3D array with broadcasting along multiple dimensions, float32
    a = np.zeros((1, 1, 1), dtype=np.float32)
    fill_value = np.array([[[1.2, 3.4]]], dtype=np.float32)
    dtype = np.dtype('float32')
    shape = (2, 2, 2)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 7: Float32 input/output with small size override
    a = np.array([1, 2], dtype=np.float32)
    fill_value = np.array([5], dtype=np.float32)
    dtype = np.dtype('float32')
    shape = (3,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 8: 1D array reshaped via shape param, float32
    a = np.ones((10,), dtype=np.float32)
    fill_value = np.array([0.5], dtype=np.float32)
    dtype = np.dtype('float32')
    shape = (2, 5)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 9: Shrinking shape via shape parameter, int32
    a = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    fill_value = np.array([10], dtype=np.int32)
    dtype = np.dtype('int32')
    shape = (1, 1)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 10: Column vector broadcasting, int32
    a = np.ones((3, 3), dtype=np.int32)
    fill_value = np.array([[1], [2], [3]], dtype=np.int32)
    dtype = np.dtype('int32')
    shape = (3, 3)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    return list_of_inputs

generated_inputs["jax.numpy.full_like_1"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_1'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_1'], lib="jax", suffix=1)
