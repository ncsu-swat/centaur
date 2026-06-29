
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with negative values
    operand = np.array([-1, 0, 1, 2], dtype=np.int32)
    reps = [3]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 2: 2D float32 array
    operand = np.random.randn(2, 3).astype(np.float32)
    reps = [2, 4]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 3: 3D int16 array
    operand = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    reps = [1, 3, 2]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 4: 4D float64 array
    operand = np.random.randn(1, 2, 1, 3).astype(np.float64)
    reps = [2, 1, 3, 1]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 5: 2D boolean array
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    reps = [3, 3]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 6: 1D large repeat float32
    operand = np.array([0.5, -0.5], dtype=np.float32)
    reps = [10]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 7: 3D complex array
    operand = (np.random.randn(2, 1, 2) + 1j * np.random.randn(2, 1, 2)).astype(np.complex64)
    reps = [2, 4, 1]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 8: 5D array with singletons
    operand = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    reps = [2, 2, 2, 2, 2]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 9: Large 2D array
    operand = np.random.randn(50, 50).astype(np.float32)
    reps = [1, 2]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 10: 3D int64 array
    operand = np.array([[[-5]], [[10]]], dtype=np.int64)
    reps = [2, 3, 4]
    list_of_inputs.append({"operand": operand, "reps": reps})

    # Input 11: 1D array tiled 0 times (empty output)
    operand = np.array([1, 2, 3], dtype=np.int32)
    reps = [0]
    list_of_inputs.append({"operand": operand, "reps": reps})

    return list_of_inputs

generated_inputs["jax.lax.tile_2"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tile_2'.")


check_valid('jax.lax.tile', generated_inputs['jax.lax.tile_2'], lib="jax", suffix=2)
