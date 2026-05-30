
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ones_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 input, float64 dtype, 2D shape
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.dtype('float64')
    shape = (2, 3)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 2: 2D int32 input with negative values, float32 dtype, 1D shape
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    dtype = np.dtype('float32')
    shape = (4,)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 3: 3D float64 input, bool dtype, 3D shape
    a = np.random.randn(2, 2, 2).astype(np.float64)
    dtype = np.dtype('bool')
    shape = (1, 2, 3)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 4: 1D bool input, complex64 dtype, 2D shape
    a = np.array([True, False, True], dtype=np.bool_)
    dtype = np.dtype('complex64')
    shape = (3, 3)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 5: Scalar-like 0D input, int32 dtype, 4D shape
    a = np.array(5.0, dtype=np.float32)
    dtype = np.dtype('int32')
    shape = (2, 2, 2, 2)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 6: 2D complex64 input, int8 dtype, 3D shape
    a = np.array([[1+2j, 3+4j]], dtype=np.complex64)
    dtype = np.dtype('int8')
    shape = (2, 1, 3)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 7: 4D int8 input, uint32 dtype, 2D shape
    a = np.ones((1, 2, 1, 2), dtype=np.int8)
    dtype = np.dtype('uint32')
    shape = (5, 5)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 8: 3D uint8 input, float16 dtype, 1D shape
    a = np.zeros((2, 2, 2), dtype=np.uint8)
    dtype = np.dtype('float16')
    shape = (8,)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 9: 2D float32 input, int64 dtype, 3D shape
    a = np.random.randn(3, 4).astype(np.float32)
    dtype = np.dtype('int64')
    shape = (2, 3, 4)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    # Input 10: 1D int16 input, complex128 dtype, 4D shape
    a = np.array([-10, 20, -30], dtype=np.int16)
    dtype = np.dtype('complex128')
    shape = (1, 1, 3, 3)
    list_of_inputs.append({"a": a, "dtype": dtype, "shape": shape})

    return list_of_inputs

generated_inputs["jax.numpy.ones_like_2"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_like_2'.")


check_valid('jax.numpy.ones_like', generated_inputs['jax.numpy.ones_like_2'], lib="jax", suffix=2)
