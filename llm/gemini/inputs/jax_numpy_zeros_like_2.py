
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: Float32 2D input, overridden to float32 1D array of shape 5
    a = np.random.randn(2, 3).astype(np.float32)
    dtype = np.dtype('float32')
    shape = 5
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 2D input, overridden to int32 1D array of shape 10
    a = np.random.randint(0, 10, (4, 4)).astype(np.int32)
    dtype = np.dtype('int32')
    shape = 10
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 3D input, overridden to float64 1D array of shape 2
    a = np.random.randn(3, 3, 3).astype(np.float64)
    dtype = np.dtype('float64')
    shape = 2
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean 1D input, overridden to bool 1D array of shape 8
    a = np.random.choice([True, False], size=(5,)).astype(np.bool_)
    dtype = np.dtype('bool')
    shape = 8
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 2D input, overridden to complex64 1D array of shape 3
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    dtype = np.dtype('complex64')
    shape = 3
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32 2D input with negative values, overridden to int16 1D array of shape 12
    a = np.random.randn(1, 5).astype(np.float32) - 5.0
    dtype = np.dtype('int16')
    shape = 12
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int64 1D input, overridden to int64 1D array of shape 20
    a = np.random.randint(-100, 100, (10,)).astype(np.int64)
    dtype = np.dtype('int64')
    shape = 20
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32 4D input, overridden to float32 1D array of shape 1
    a = np.random.rand(2, 3, 4, 5).astype(np.float32)
    dtype = np.dtype('float32')
    shape = 1
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint8 2D input, overridden to uint8 1D array of shape 15
    a = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
    dtype = np.dtype('uint8')
    shape = 15
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float16 5D input, overridden to float16 1D array of shape 4
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float16)
    dtype = np.dtype('float16')
    shape = 4
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_like_2"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_2'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_2'], lib="jax", suffix=2)
