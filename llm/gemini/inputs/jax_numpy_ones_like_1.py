
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ones_like_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(5).astype(np.float32)
    dtype = np.float32
    shape = 5
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 2
    a = np.random.randint(-10, 10, size=(2, 3)).astype(np.int32)
    dtype = np.int32
    shape = 3
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 3
    a = np.random.randn(2, 2, 2).astype(np.float64)
    dtype = np.float64
    shape = 10
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 4
    a = np.random.choice([True, False], size=(4,)).astype(np.bool_)
    dtype = np.bool_
    shape = 2
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 5
    a = np.random.randint(0, 100, size=(1, 2, 3, 4)).astype(np.int64)
    dtype = np.float32
    shape = 8
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 6
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    dtype = np.complex64
    shape = 4
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 7
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    dtype = np.int32
    shape = 6
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 8
    a = np.array(42.0, dtype=np.float32)
    dtype = np.float32
    shape = 1
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 9
    a = np.random.randint(0, 255, size=(2, 2, 1)).astype(np.uint8)
    dtype = np.uint8
    shape = 12
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

    # Input 10
    a = np.random.randint(-5, 5, size=(4, 4)).astype(np.int16)
    dtype = np.int16
    shape = 7
    list_of_inputs.append({"a": copy.deepcopy(a), "dtype": dtype, "shape": shape})

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
