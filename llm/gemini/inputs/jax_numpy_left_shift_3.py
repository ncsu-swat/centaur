
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def left_shift_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 shift amounts
    list_of_inputs.append({
        'x': 1,
        'y': np.array([1, 2, 3], dtype=np.int32)
    })

    # Input 2: 2D int32 shift amounts
    list_of_inputs.append({
        'x': 5,
        'y': np.array([[0, 1], [2, 3]], dtype=np.int32)
    })

    # Input 3: Negative integer for x, 1D int64 shift amounts
    list_of_inputs.append({
        'x': -2,
        'y': np.array([1, 2, 4], dtype=np.int64)
    })

    # Input 4: 3D int32 shift amounts
    list_of_inputs.append({
        'x': 10,
        'y': np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    })

    # Input 5: 3D array of ones with int32 type
    list_of_inputs.append({
        'x': 3,
        'y': np.ones((2, 2, 2), dtype=np.int32)
    })

    # Input 6: x is 0, shifting shouldn't change value, int32 shift amounts
    list_of_inputs.append({
        'x': 0,
        'y': np.array([4, 5, 6], dtype=np.int32)
    })

    # Input 7: Negative boundary value, int64 shift amounts
    list_of_inputs.append({
        'x': -128,
        'y': np.array([0, 1, 2], dtype=np.int64)
    })

    # Input 8: 1D arange shift amounts, int32
    list_of_inputs.append({
        'x': 100,
        'y': np.arange(5, dtype=np.int32)
    })

    # Input 9: Column vector (2D) shift amounts, int64
    list_of_inputs.append({
        'x': 42,
        'y': np.array([[1], [2], [3]], dtype=np.int64)
    })

    # Input 10: 2D array of random shift values up to 5, int32
    list_of_inputs.append({
        'x': 7,
        'y': np.random.randint(0, 5, size=(4, 4), dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.left_shift_3"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_3'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_3'], lib="jax", suffix=3)
