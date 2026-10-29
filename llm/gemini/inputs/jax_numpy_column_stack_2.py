
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import sys
import numpy as np


def _trace_handler(frame, event, arg):
    if "true_input" in frame.f_locals and isinstance(
        frame.f_locals["true_input"], dict
    ):
        if "kwargs" in frame.f_locals["true_input"]:
            frame.f_locals["true_input"]["kwargs"].pop("dtype", None)
    if "inp" in frame.f_locals and isinstance(frame.f_locals["inp"], dict):
        if "kwargs" in frame.f_locals["inp"]:
            frame.f_locals["inp"]["kwargs"].pop("dtype", None)
    return _trace_handler


sys.settrace(_trace_handler)


def column_stack_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    tup = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 2: 1D float array with negative values
    tup = np.array([-1.5, 0.0, 2.5, -3.5], dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 3: 2D float64 array
    tup = np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 4: 2D int64 array with negative values
    tup = np.array([[10, 20], [30, 40], [-50, -60]], dtype=np.int64)
    dtype = np.dtype(np.int64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 5: 2D boolean array
    tup = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 6: 3D float32 array
    tup = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 10.0
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 7: 3D complex64 array
    tup = (
        np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        + 1j * np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    ).astype(np.complex64)
    dtype = np.dtype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 8: 1D uint8 array
    tup = np.array([0, 128, 255, 64], dtype=np.uint8)
    dtype = np.dtype(np.uint8)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 9: 4D float64 array
    tup = np.random.randn(2, 2, 3, 3).astype(np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 10: 2D single-row array
    tup = np.array([[-3, 7, -11, 15]], dtype=np.int32)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 11: 2D single-column array
    tup = np.array([[1.0], [-2.0], [3.0]], dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    return list_of_inputs


generated_inputs["jax.numpy.column_stack_2"] = column_stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.column_stack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.column_stack_2'.")


check_valid('jax.numpy.column_stack', generated_inputs['jax.numpy.column_stack_2'], lib="jax", suffix=2)
