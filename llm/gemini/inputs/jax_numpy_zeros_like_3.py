
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeros_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, float32 dtype, 1D list shape
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = [4]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D int32 array, int32 dtype, 2D list shape
    a = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = [2, 3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 3D bool array, bool dtype, 3D list shape
    a = np.ones((2, 2, 2), dtype=bool)
    dtype = np.dtype(np.bool_)
    shape = [2, 2, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: 0D scalar array, float32 dtype, scalar shape
    a = np.array(-5.5, dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = []
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: 4D int32 array, int32 dtype, 4D list shape
    a = np.zeros((1, 2, 1, 3), dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = [1, 2, 1, 3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 1D float32 array with negative values, bool dtype, 1D list shape
    a = np.array([-10.0, -20.5, 30.2, 40.8], dtype=np.float32)
    dtype = np.dtype(np.bool_)
    shape = [10]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: 2D int32 array, float32 dtype, 2D list shape
    a = np.array([[1, -2], [3, -4]], dtype=np.int32)
    dtype = np.dtype(np.float32)
    shape = [3, 5]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: 2D bool array, int32 dtype, 2D list shape
    a = np.array([[True, False], [False, True]], dtype=bool)
    dtype = np.dtype(np.int32)
    shape = [4, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: 3D float32 array, float32 dtype, 3D list shape
    a = np.ones((2, 1, 3), dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = [1, 3, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: 2D int32 array, int32 dtype, shape with zero dimension
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = [0, 5]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 11: 5D float32 array, float32 dtype, 5D list shape
    a = np.ones((1, 1, 2, 2, 1), dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = [1, 1, 2, 2, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

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
