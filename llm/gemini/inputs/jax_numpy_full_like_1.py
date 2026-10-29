
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 0D fill_value, float32
    a = np.zeros(5, dtype=np.float32)
    fill_value = np.array(2.5, dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = (5,)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 2: 2D array, 0D negative fill_value, int32
    a = np.ones((2, 3), dtype=np.int32)
    fill_value = np.array(-7, dtype=np.int32)
    dtype = np.dtype(np.int32)
    shape = (3, 4)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 3: 3D array, 1D broadcastable fill_value, float64
    a = np.empty((2, 2, 2), dtype=np.float64)
    fill_value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    dtype = np.dtype(np.float64)
    shape = (2, 3, 4)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 4: 0D scalar array, 0D boolean fill_value, bool
    a = np.array(1, dtype=np.int32)
    fill_value = np.array(True, dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    shape = ()
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 5: 2D array, matching 2D fill_value, int64
    a = np.zeros((2, 2), dtype=np.int64)
    fill_value = np.array([[10, 20], [30, 40]], dtype=np.int64)
    dtype = np.dtype(np.int64)
    shape = (2, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 6: 4D array, 0D fill_value, float16
    a = np.ones((1, 2, 1, 3), dtype=np.float16)
    fill_value = np.array(-0.5, dtype=np.float16)
    dtype = np.dtype(np.float16)
    shape = (1, 2, 1, 3)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 7: 1D array, 2D broadcastable fill_value, int16
    a = np.arange(6, dtype=np.int16)
    fill_value = np.array([[100, 200, 300]], dtype=np.int16)
    dtype = np.dtype(np.int16)
    shape = (4, 3)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 8: 2D array, 0D complex fill_value, complex64
    a = np.zeros((2, 3), dtype=np.complex64)
    fill_value = np.array(1.0 - 2.0j, dtype=np.complex64)
    dtype = np.dtype(np.complex64)
    shape = (2, 5)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 9: 3D array, 0D fill_value, uint8
    a = np.ones((1, 4, 1), dtype=np.uint8)
    fill_value = np.array(255, dtype=np.uint8)
    dtype = np.dtype(np.uint8)
    shape = (3, 1, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 10: 1D array, 0D negative fill_value, int8
    a = np.zeros(3, dtype=np.int8)
    fill_value = np.array(-128, dtype=np.int8)
    dtype = np.dtype(np.int8)
    shape = (10,)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    # Input 11: 2D array, 0D infinity fill_value, float32
    a = np.zeros((4, 4), dtype=np.float32)
    fill_value = np.array(np.inf, dtype=np.float32)
    dtype = np.dtype(np.float32)
    shape = (3, 3)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "fill_value": fill_value, "dtype": dtype, "shape": shape}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.full_like_1"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_1'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_1'], lib="jax", suffix=1)
