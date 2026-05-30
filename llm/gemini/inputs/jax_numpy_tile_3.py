
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 1 repetition
    A = np.array([1, 2, 3], dtype=np.int32)
    reps = [2]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 2: 1D array, 2 repetitions (A will be promoted to 2D)
    A = np.array([4, 5], dtype=np.int32)
    reps = [2, 3]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 3: 2D array, 2 repetitions
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    reps = [2, 1]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 4: 2D array, 3 repetitions (promoted to 3D)
    A = np.array([[1.5, 2.5]], dtype=np.float64)
    reps = [3, 2, 1]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 5: 3D random array
    A = np.random.randn(2, 3, 4).astype(np.float32)
    reps = [1, 2, 1]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 6: 3D int array, shorter reps list (will be padded on the left)
    A = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int64)
    reps = [2]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 7: Boolean array
    A = np.array([[True, False], [False, True]], dtype=np.bool_)
    reps = [2, 2]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 8: 4D array with diverse shapes
    A = np.ones((1, 2, 1, 3), dtype=np.float32)
    reps = [2, 1, 3, 1]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 9: Zero repetitions (creates empty array)
    A = np.array([10, 20], dtype=np.int32)
    reps = [0]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    # Input 10: 2D array, larger repetition factors
    A = np.array([[5]], dtype=np.int16)
    reps = [10, 10]
    list_of_inputs.append({"A": copy.deepcopy(A), "reps": reps})

    return list_of_inputs

generated_inputs["jax.numpy.tile_3"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tile_3'.")


check_valid('jax.numpy.tile', generated_inputs['jax.numpy.tile_3'], lib="jax", suffix=3)
