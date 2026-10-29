
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D array with 1D reps
    A = np.array([1, -2, 3], dtype=np.int32)
    reps = [3]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 2: 1D array tiled into 2D with reps longer than ndim
    A = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    reps = [2, 3]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 3: 2D array tiled along both dimensions
    A = np.array([[1, 2], [3, 4]], dtype=np.int64)
    reps = [2, 1]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 4: 2D float array tiled with 1D reps (promoted along trailing dimension)
    A = np.random.randn(3, 4).astype(np.float64)
    reps = [2]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 5: 2D array tiled to 3D
    A = np.array([[True, False], [False, True]], dtype=np.bool_)
    reps = [2, 2, 2]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 6: 0D scalar array tiled to 1D
    A = np.array(-5, dtype=np.int32)
    reps = [4]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 7: 3D complex array
    A = (np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)).astype(
        np.complex64
    )
    reps = [1, 2, 1]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 8: reps containing 0 (empty result)
    A = np.array([[10, 20], [30, 40]], dtype=np.int16)
    reps = [0, 2]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 9: 4D float array with identical reps
    A = np.ones((2, 1, 3, 2), dtype=np.float32)
    reps = [1, 3, 1, 2]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 10: 1D array tiled into 3D
    A = np.array([0, 1], dtype=np.uint8)
    reps = [2, 3, 4]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 11: 0D scalar array tiled to 2D
    A = np.array(3.14159, dtype=np.float32)
    reps = [2, 3]
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

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
