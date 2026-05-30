
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.float32
    shape = [3]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, overriding shape and dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = np.float64
    shape = [3, 4]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array, overriding to complex
    a = np.array([[[True, False]]], dtype=np.bool_)
    dtype = np.complex64
    shape = [2, 2, 2]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with negative values
    a = np.array([-1.5, -2.5, -3.5], dtype=np.float64)
    dtype = np.int32
    shape = [5]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar (0D array)
    a = np.array(5, dtype=np.int16)
    dtype = np.float32
    shape = [1]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger multi-dimensional float16 array
    a = np.random.randn(2, 3, 4).astype(np.float16)
    dtype = np.float16
    shape = [2, 3, 4]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Unsigned int array, overriding to larger uint
    a = np.array([10, 20], dtype=np.uint8)
    dtype = np.uint32
    shape = [10]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array, changing dimensions
    a = np.empty((0, 5), dtype=np.float32)
    dtype = np.float32
    shape = [0, 10]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex array
    a = np.array([1+2j, 3+4j], dtype=np.complex128)
    dtype = np.complex128
    shape = [2]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int8 array, downsizing shape
    a = np.zeros((100, 100), dtype=np.int8)
    dtype = np.int8
    shape = [5, 5]
    input_dict = {"a": a, "dtype": dtype, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.zeros_like_3"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_3'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_3'], lib="jax", suffix=3)
