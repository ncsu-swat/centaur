
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 1D reps
    A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reps = (2,)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 2: 2D float32 array, 2D reps
    A = np.random.randn(2, 3).astype(np.float32)
    reps = (2, 2)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 3: 1D int32 array, 2D reps (reps has more dimensions)
    A = np.array([1, 2, 3], dtype=np.int32)
    reps = (2, 3)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 4: 3D float32 array, 3D reps
    A = np.random.randn(2, 1, 3).astype(np.float32)
    reps = (1, 4, 2)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 5: 2D boolean array, 2D reps
    A = np.array([[True, False], [False, True]], dtype=bool)
    reps = (3, 1)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 6: 4D float64 array, 4D reps
    A = np.random.randn(1, 2, 1, 2).astype(np.float64)
    reps = (2, 1, 3, 2)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 7: 0D array (scalar), 1D reps
    A = np.array(5.0, dtype=np.float32)
    reps = (4,)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 8: 3D int16 array, 2D reps (reps has fewer dimensions)
    A = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int16)
    reps = (2, 3)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 9: 2D float32 array with zero-size dimension, 2D reps
    A = np.empty((0, 3), dtype=np.float32)
    reps = (2, 2)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 10: 1D complex64 array, 1D reps
    A = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    reps = (3,)
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    return list_of_inputs

generated_inputs["jax.numpy.tile_2"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tile_2'.")


check_valid('jax.numpy.tile', generated_inputs['jax.numpy.tile_2'], lib="jax", suffix=2)
