
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32 -> float32
    tup = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 2: 1D array with negative values
    tup = np.array([-5.0, 0.0, 5.0, 10.5], dtype=np.float32)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 3: 2D array, float32 -> float32
    tup = np.random.randn(3, 4).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 4: 2D array, int64 -> int32
    tup = np.arange(12, dtype=np.int64).reshape(4, 3)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 5: 3D array, float64
    tup = np.random.randn(2, 3, 5).astype(np.float64)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 6: 3D array, uint8 -> int32
    tup = np.ones((4, 2, 2), dtype=np.uint8)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 7: 4D array, float32
    tup = np.random.randn(2, 2, 3, 4).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 8: 1D array of booleans -> bool
    tup = np.array([True, False, True, True], dtype=np.bool_)
    dtype = np.bool_
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 9: 2D complex numbers
    tup = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    dtype = np.complex64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 10: 5D array
    tup = np.zeros((1, 2, 1, 3, 2), dtype=np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 11: 2D array, boolean -> int64
    tup = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.int64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    return list_of_inputs

generated_inputs["jax.numpy.vstack_2"] = vstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vstack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vstack_2'.")


check_valid('jax.numpy.vstack', generated_inputs['jax.numpy.vstack_2'], lib="jax", suffix=2)
