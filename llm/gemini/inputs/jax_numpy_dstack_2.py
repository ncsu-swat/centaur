
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dstack_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    tup = np.array([1, -2, 3, -4], dtype=np.int32)
    dtype = np.dtype(np.int64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 2: 2D float32 array
    tup = np.random.randn(2, 3).astype(np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 3: 2D float64 array with negative and positive values
    tup = np.array([[-1.5, 2.0, -3.2], [4.1, -5.0, 6.3]], dtype=np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 4: 3D float32 array
    tup = np.random.randn(3, 4, 5).astype(np.float32)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 5: 3D integer array
    tup = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 6: 4D float32 array
    tup = np.random.randn(2, 2, 3, 4).astype(np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 7: 2D boolean array
    tup = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 8: 1D complex array
    tup = np.array([1.0 + 2.0j, -3.0 - 4.0j, 5.0 + 0.0j], dtype=np.complex64)
    dtype = np.dtype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 9: 1D float64 array from linspace
    tup = np.linspace(-5.0, 5.0, 8, dtype=np.float64)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 10: 3D zeros array of int16
    tup = np.zeros((4, 2, 3), dtype=np.int16)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 11: 2D uint8 array
    tup = np.full((3, 5), 200, dtype=np.uint8)
    dtype = np.dtype(np.uint16)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    return list_of_inputs


generated_inputs["jax.numpy.dstack_2"] = dstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dstack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dstack_2'.")


check_valid('jax.numpy.dstack', generated_inputs['jax.numpy.dstack_2'], lib="jax", suffix=2)
