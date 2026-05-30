
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def full_like_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, fill with True, convert to float32, change shape
    a = np.random.randn(2, 2).astype(np.float32)
    fill_value = True
    dtype = np.dtype('float32')
    shape = (3, 3)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 2: 3D int32 array, fill with False, convert to int64, smaller shape
    a = np.random.randint(-10, 10, size=(3, 4, 5)).astype(np.int32)
    fill_value = False
    dtype = np.dtype('int64')
    shape = (2, 2)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 3: 1D bool array, fill with True, convert to bool, larger shape
    a = np.array([True, False, True], dtype=bool)
    fill_value = True
    dtype = np.dtype('bool')
    shape = (10,)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 4: 2D float64 array, fill with False, convert to float64, reshaped
    a = np.random.rand(1, 5).astype(np.float64)
    fill_value = False
    dtype = np.dtype('float64')
    shape = (5, 1)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 5: 4D int16 array, fill with True, convert to int16, reduce dimension
    a = np.zeros((1, 2, 3, 4), dtype=np.int16)
    fill_value = True
    dtype = np.dtype('int16')
    shape = (2, 3, 4)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 6: 0D float32 array, fill with True, convert to float32, 1D shape
    a = np.array(1.5, dtype=np.float32)
    fill_value = True
    dtype = np.dtype('float32')
    shape = (1,)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 7: 2D float32 array, fill with False, convert to complex64, different shape
    a = np.random.randn(5, 5).astype(np.float32)
    fill_value = False
    dtype = np.dtype('complex64')
    shape = (2, 5)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 8: 2D uint8 array, fill with True, convert to uint8, 3D shape
    a = np.array([[1]], dtype=np.uint8)
    fill_value = True
    dtype = np.dtype('uint8')
    shape = (4, 4, 4)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 9: 4D float32 array, fill with False, convert to float16, 3D shape
    a = np.random.rand(2, 3, 2, 3).astype(np.float32)
    fill_value = False
    dtype = np.dtype('float16')
    shape = (1, 1, 1)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    # Input 10: 1D negative int32 array, fill with True, convert to int32, larger 1D shape
    a = np.array([-1, -2, -3], dtype=np.int32)
    fill_value = True
    dtype = np.dtype('int32')
    shape = (6,)
    list_of_inputs.append({"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.full_like_4"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_4'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_4'], lib="jax", suffix=4)
