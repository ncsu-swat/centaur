
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ones_like_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = 4
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 2: 2D int32 array with shape 0 (empty array)
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 3: 0D scalar float32 array with shape 1
    a = np.array(-5.5, dtype=np.float32)
    dtype = np.dtype(np.int32)
    shape = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 4: 3D float32 array changing to bool dtype
    a = np.random.randn(2, 3, 4).astype(np.float32)
    dtype = np.dtype(np.bool_)
    shape = 6
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 5: Boolean array to complex64 dtype
    a = np.array([True, False, True], dtype=np.bool_)
    dtype = np.dtype(np.complex64)
    shape = 8
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 6: 4D int32 array to uint8 dtype
    a = np.ones((1, 2, 2, 3), dtype=np.int32)
    dtype = np.dtype(np.uint8)
    shape = 12
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 7: Float32 with negative values
    a = np.array([-10.0, -20.0, 0.0, 30.0], dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = 2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 8: Complex64 array to int16 dtype
    a = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    dtype = np.dtype(np.int16)
    shape = 5
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 9: 2D int8 array to int32 dtype
    a = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    dtype = np.dtype(np.int32)
    shape = 10
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 10: 2D uint8 array to float32 dtype
    a = np.eye(4, dtype=np.uint8)
    dtype = np.dtype(np.float32)
    shape = 16
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    # Input 11: 3D boolean array to bool dtype
    a = np.zeros((2, 2, 2), dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    shape = 3
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "shape": shape})
    )

    return list_of_inputs


generated_inputs["jax.numpy.ones_like_1"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_like_1'.")


check_valid('jax.numpy.ones_like', generated_inputs['jax.numpy.ones_like_1'], lib="jax", suffix=1)
