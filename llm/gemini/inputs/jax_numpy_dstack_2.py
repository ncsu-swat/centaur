
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32
    tup = np.random.randn(5).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 2: 2D array, float32
    tup = np.random.randn(2, 3).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 3: 3D array, float64
    tup = np.random.randn(2, 3, 4).astype(np.float64)
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 4: 4D array, int32
    tup = np.random.randint(-10, 10, (2, 2, 2, 2)).astype(np.int32)
    dtype = np.int32
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 5: 2D array, negative values, int16
    tup = np.random.randint(-100, -10, (3, 5)).astype(np.int16)
    dtype = np.int16
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 6: 3D array, bool
    tup = np.random.choice([True, False], size=(2, 2, 3))
    dtype = np.bool_
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 7: 1D array, uint8
    tup = np.random.randint(0, 255, (10,)).astype(np.uint8)
    dtype = np.uint8
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 8: 3D array, float16
    tup = np.random.randn(2, 4, 3).astype(np.float16)
    dtype = np.float16
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 9: 2D array, int64
    tup = np.random.randint(0, 100, (4, 4)).astype(np.int64)
    dtype = np.int64
    list_of_inputs.append(copy.deepcopy({"tup": tup, "dtype": dtype}))

    # Input 10: 4D array, float32
    tup = np.random.randn(3, 1, 2, 2).astype(np.float32)
    dtype = np.float32
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
