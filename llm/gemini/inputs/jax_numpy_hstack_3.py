
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hstack_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    tup = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 2: 2D int32 array converted to int64
    tup = np.array([[5, -2, 3], [-1, 4, 0]], dtype=np.int32)
    dtype = np.dtype(np.int64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 3: 1D float64 array
    tup = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 4: 3D float32 array
    tup = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 5: 3D array with negative values
    tup = np.random.uniform(-10.0, 10.0, size=(3, 4, 2)).astype(np.float64)
    dtype = np.dtype(np.float64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 6: 2D boolean array
    tup = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 7: 4D int16 array cast to float32
    tup = np.random.randint(-100, 100, size=(2, 3, 2, 4)).astype(np.int16)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 8: 1D int32 array with negative integers
    tup = np.array([-10, -20, 30, -40, 50], dtype=np.int32)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 9: 2D complex64 array
    tup = np.array([[1.0 + 2.0j, -3.0 + 0.5j], [0.0 - 1.0j, 4.0 + 2.5j]], dtype=np.complex64)
    dtype = np.dtype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 10: 3D uint8 array cast to int32
    tup = np.random.randint(0, 255, size=(4, 2, 3)).astype(np.uint8)
    dtype = np.dtype(np.int32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 11: 2D single row array
    tup = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    dtype = np.dtype(np.float32)
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    return list_of_inputs

generated_inputs["jax.numpy.hstack_3"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hstack_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hstack_3'.")


check_valid('jax.numpy.hstack', generated_inputs['jax.numpy.hstack_3'], lib="jax", suffix=3)
