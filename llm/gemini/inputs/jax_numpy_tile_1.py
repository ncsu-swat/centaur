
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    A = np.array([1, 2, 3], dtype=np.int32)
    reps = 2
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 2: 1D float array with negative values
    A = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    reps = 3
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 3: 2D float64 array
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    reps = 2
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 4: 2D array with reps=1 (identity tiling)
    A = np.arange(6, dtype=np.int64).reshape(2, 3)
    reps = 1
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 5: 3D boolean array
    A = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    reps = 4
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 6: 0D scalar array
    A = np.array(42.0, dtype=np.float32)
    reps = 3
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 7: 4D complex array
    A = np.ones((2, 1, 3, 2), dtype=np.complex64)
    reps = 2
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 8: reps = 0 resulting in zero-size dimension
    A = np.array([10, 20, 30], dtype=np.int32)
    reps = 0
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 9: 2D int16 array with large reps
    A = np.array([[-5, -10], [15, 20]], dtype=np.int16)
    reps = 5
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 10: 1D empty array
    A = np.empty((0,), dtype=np.float32)
    reps = 3
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 11: 3D uint8 array
    A = np.random.randint(0, 255, size=(2, 3, 4), dtype=np.uint8)
    reps = 2
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    return list_of_inputs

generated_inputs["jax.numpy.tile_1"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tile_1'.")


check_valid('jax.numpy.tile', generated_inputs['jax.numpy.tile_1'], lib="jax", suffix=1)
