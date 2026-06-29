
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers, small rep
    operand = np.array([1, 2, 3], dtype=np.int32)
    reps = (3,)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 2: 2D array of floats, medium reps
    operand = np.random.randn(2, 3).astype(np.float32)
    reps = (2, 4)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 3: 3D array of float64, with reps=1 on some dimensions
    operand = np.random.randn(2, 2, 2).astype(np.float64)
    reps = (1, 3, 2)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 4: 1D boolean array
    operand = np.array([True, False, True], dtype=bool)
    reps = (5,)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 5: 2D array of int64 containing negative numbers
    operand = np.array([[-1, -2], [3, 4]], dtype=np.int64)
    reps = (3, 1)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 6: 4D array, all reps are 1 (no-op tiling)
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    reps = (1, 1, 1, 1)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 7: 3D array with zeros and negative values
    operand = np.array([[[0, -1], [2, -3]], [[4, 5], [-6, 7]]], dtype=np.int32)
    reps = (2, 1, 2)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 8: 2D float32 array, large reps
    operand = np.random.randn(3, 3).astype(np.float32)
    reps = (10, 5)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 9: 1D array with single element
    operand = np.array([42], dtype=np.int16)
    reps = (10,)
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 10: 5D array, float32, various reps
    operand = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    reps = (2, 1, 3, 1, 4)
    list_of_inputs.append({"operand": operand, "reps": reps})

    return list_of_inputs

generated_inputs["jax.lax.tile_1"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tile_1'.")


check_valid('jax.lax.tile', generated_inputs['jax.lax.tile_1'], lib="jax", suffix=1)
