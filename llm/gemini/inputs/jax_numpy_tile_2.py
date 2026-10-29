
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, 1D reps
    A = np.array([1, 2, 3], dtype=np.int32)
    reps = (2,)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 2: 2D float array, 2D reps
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reps = (2, 3)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 3: 0D scalar array, 1D reps
    A = np.array(5.5, dtype=np.float64)
    reps = (4,)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 4: 1D array promoted to 2D via reps
    A = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    reps = (3, 2)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 5: 3D int array, 3D reps
    A = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    reps = (1, 2, 1)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 6: 2D float array with negative values
    A = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    reps = (2, 1)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 7: 2D int array, asymmetric reps
    A = np.array([[5, -6], [7, 8]], dtype=np.int32)
    reps = (1, 3)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 8: 4D float array, 4D reps
    A = np.ones((1, 2, 1, 3), dtype=np.float32)
    reps = (2, 1, 3, 1)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 9: 2D array tiled with zero repetition along one axis
    A = np.array([[10, 20], [-10, -20]], dtype=np.int32)
    reps = (0, 2)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 10: 3D array with reps shorter than array ndim
    A = np.random.randn(2, 3, 4).astype(np.float64)
    reps = (2,)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

    # Input 11: 1D array promoted to 3D via reps
    A = np.array([42], dtype=np.int32)
    reps = (2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"A": A, "reps": reps}))

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
