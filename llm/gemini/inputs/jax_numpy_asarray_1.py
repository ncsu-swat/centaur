
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, 2.5, -3.7, 4.2], dtype=np.float32)
    dtype = np.dtype(np.float32)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 2: 2D int32 array with negative values
    a = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int32)
    dtype = np.dtype(np.int32)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 3: 3D float64 array
    a = np.random.randn(2, 3, 4).astype(np.float64)
    dtype = np.dtype(np.float64)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 4: 0D scalar array
    a = np.array(42, dtype=np.int64)
    dtype = np.dtype(np.int64)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 5: 2D boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 6: 4D uint8 array
    a = np.arange(24, dtype=np.uint8).reshape((2, 3, 2, 2))
    dtype = np.dtype(np.uint8)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 7: 1D complex64 array
    a = np.array([1.0 + 2.0j, -3.0 - 4.0j, 0.5j], dtype=np.complex64)
    dtype = np.dtype(np.complex64)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 8: Empty 1D float32 array
    a = np.array([], dtype=np.float32)
    dtype = np.dtype(np.float32)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 9: 2D float16 array
    a = np.array([[0.1, -0.2], [100.5, -200.25]], dtype=np.float16)
    dtype = np.dtype(np.float16)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 10: 3D int16 array with positive and negative values
    a = np.array([[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]], dtype=np.int16)
    dtype = np.dtype(np.int16)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 11: 2D complex128 array
    a = np.array(
        [[1.5 - 2.5j, 3.0 + 0.0j], [-4.2 + 1.1j, 0.0 - 5.0j]],
        dtype=np.complex128,
    )
    dtype = np.dtype(np.complex128)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    # Input 12: Empty 3D int32 array with zero dimension
    a = np.empty((2, 0, 3), dtype=np.int32)
    dtype = np.dtype(np.int32)
    order = "K"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "dtype": dtype, "order": order, "copy": copy_flag})
    )

    return list_of_inputs


generated_inputs["jax.numpy.asarray_1"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_1'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_1'], lib="jax", suffix=1)
