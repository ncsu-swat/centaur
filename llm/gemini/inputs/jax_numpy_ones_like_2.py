
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ones_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array to 2D float32
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.dtype(np.float32)
    shape = (2, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D array to 1D int32
    a = np.array([[-1.5, 2.3], [0.0, -4.7]], dtype=np.float32)
    dtype = np.dtype(np.int32)
    shape = (4,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 0D scalar array to 3D bool
    a = np.array(5, dtype=np.int32)
    dtype = np.dtype(np.bool_)
    shape = (2, 3, 4)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: 3D array, same shape, float32
    a = np.ones((2, 2, 3), dtype=np.int32)
    dtype = np.dtype(np.float32)
    shape = (2, 2, 3)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: 4D array to 0D scalar shape, int32
    a = np.zeros((1, 2, 3, 4), dtype=np.float32)
    dtype = np.dtype(np.int32)
    shape = ()
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 1D boolean array to 2D bool
    a = np.array([True, False, True], dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    shape = (3, 1)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: 2D array with negative values to 1D float32
    a = np.array([[-10, -20], [-30, -40]], dtype=np.int32)
    dtype = np.dtype(np.float32)
    shape = (4,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: 1D array to 4D int32
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = (1, 2, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: 2D int array to 2D bool
    a = np.arange(6, dtype=np.int32).reshape((2, 3))
    dtype = np.dtype(np.bool_)
    shape = (5, 5)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: 3D float array to 1D float32
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = (8,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

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
