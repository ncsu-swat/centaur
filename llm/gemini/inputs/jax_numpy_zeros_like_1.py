
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with float32 dtype and 1D shape
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = (5,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D int32 array with int64 dtype and 2D shape
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = np.dtype(np.int64)
    shape = (3, 4)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 3D float64 array with bool dtype and 3D shape
    a = np.ones((2, 3, 4), dtype=np.float64)
    dtype = np.dtype(np.bool_)
    shape = (2, 2, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: 0D scalar array with complex64 dtype and 1D shape
    a = np.array(42.0, dtype=np.float32)
    dtype = np.dtype(np.complex64)
    shape = (1,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: 4D bool array with uint8 dtype and 4D shape
    a = np.array([[[[True, False]]]], dtype=np.bool_)
    dtype = np.dtype(np.uint8)
    shape = (1, 2, 1, 3)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 1D int16 array with negative values, float16 dtype and 1D shape
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int16)
    dtype = np.dtype(np.float16)
    shape = (10,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: 2D complex128 array with complex128 dtype and 2D shape
    a = np.array([[1.0 + 2.0j, -3.0j]], dtype=np.complex128)
    dtype = np.dtype(np.complex128)
    shape = (4, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: 3D int8 array with int32 dtype and 3D shape
    a = np.zeros((1, 1, 1), dtype=np.int8)
    dtype = np.dtype(np.int32)
    shape = (3, 1, 5)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: 5D float32 array with float64 dtype and 5D shape
    a = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    dtype = np.dtype(np.float64)
    shape = (2, 1, 2, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: Empty 1D array with int32 dtype and 1D empty shape
    a = np.array([], dtype=np.float32)
    dtype = np.dtype(np.int32)
    shape = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 11: 2D array with float32 dtype and 0D scalar shape
    a = np.ones((2, 2), dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = ()
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.zeros_like_1"] = zeros_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.zeros_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.zeros_like_1'.")


check_valid('jax.numpy.zeros_like', generated_inputs['jax.numpy.zeros_like_1'], lib="jax", suffix=1)
