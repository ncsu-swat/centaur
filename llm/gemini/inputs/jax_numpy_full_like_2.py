
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def full_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive float fill, simple 1D shape
    a = np.arange(5, dtype=np.float32)
    fill_value = 3.5
    dtype = np.dtype('float32')
    shape = (5,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 2: 2D array, int32, negative float fill, 2D shape
    a = np.zeros((3, 3), dtype=np.int32)
    fill_value = -1.0
    dtype = np.dtype('int32')
    shape = (2, 4)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 3: 3D array, float64, zero fill, 3D shape
    a = np.ones((2, 2, 2), dtype=np.float64)
    fill_value = 0.0
    dtype = np.dtype('float64')
    shape = (1, 2, 3)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 4: 1D array, bool, large float fill (will cast to True), 1D shape
    a = np.array([True, False, True], dtype=bool)
    fill_value = 100.5
    dtype = np.dtype('bool')
    shape = (4,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 5: 4D array, complex64, small float fill, 4D shape
    a = np.random.randn(2, 2, 2, 2).astype(np.complex64)
    fill_value = 1e-5
    dtype = np.dtype('complex64')
    shape = (2, 1, 2, 1)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 6: 2D array, float16, positive float fill, 1D shape
    a = np.empty((4, 4), dtype=np.float32)
    fill_value = 0.125
    dtype = np.dtype('float16')
    shape = (8,)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 7: 0D array, int16, float fill, 0D shape
    a = np.array(5, dtype=np.int32)
    fill_value = 2.0
    dtype = np.dtype('int16')
    shape = ()
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 8: 3D array, float32, large negative float fill, 3D shape
    a = np.ones((2, 3, 4), dtype=np.float32)
    fill_value = -999.99
    dtype = np.dtype('float32')
    shape = (3, 3, 3)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 9: 5D array, int64, float fill, 5D shape
    a = np.ones((1, 1, 1, 1, 1), dtype=np.int64)
    fill_value = 42.42
    dtype = np.dtype('int64')
    shape = (2, 2, 2, 2, 2)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    # Input 10: 2D array, float64, huge float fill, 2D shape
    a = np.random.rand(5, 5).astype(np.float64)
    fill_value = 1e300
    dtype = np.dtype('float64')
    shape = (10, 10)
    list_of_inputs.append({
        "a": a,
        "fill_value": fill_value,
        "dtype": dtype,
        "shape": shape
    })

    return list_of_inputs

generated_inputs["jax.numpy.full_like_2"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_2'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_2'], lib="jax", suffix=2)
