
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vstack_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with float32 dtype conversion
    tup = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 2: 2D float array with float64 dtype conversion
    tup = np.random.randn(3, 4).astype(np.float32)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 3: 2D array with negative values and float32 dtype
    tup = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 4: 3D array with int32 conversion
    tup = np.random.randn(2, 3, 4).astype(np.float32)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 5: 1D float array with complex64 conversion
    tup = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    dtype = np.complex64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 6: 4D array with float16 conversion
    tup = np.random.randn(2, 2, 3, 3).astype(np.float32)
    dtype = np.float16
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 7: Boolean array with bool dtype
    tup = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.bool_
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 8: 1D array with int64 conversion
    tup = np.array([10, 20, 30], dtype=np.int32)
    dtype = np.int64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 9: Large 2D array with uint32 conversion
    tup = np.random.randint(0, 100, size=(5, 5)).astype(np.int32)
    dtype = np.uint32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 10: 3D float array with float64 conversion
    tup = np.random.randn(4, 2, 2).astype(np.float32)
    dtype = np.float64
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
