
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def degrees_inputs():
    list_of_inputs = []

    # Input 1: 0-dimensional scalar float32 array
    x = np.array(np.pi, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, and zero values
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64
    x = np.array(
        [[0.0, np.pi / 4, np.pi / 2], [np.pi, 3 * np.pi / 2, 2 * np.pi]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array with random normal values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array with float16
    x = (np.random.rand(2, 2, 3, 3) * 2 * np.pi).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D integer array (int32)
    x = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D integer array (int64)
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with special values (inf, -inf, nan)
    x = np.array([-np.inf, np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: High-dimensional array (5D)
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * (np.pi / 6)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Array with very large values
    x = np.array([1e6, -1e6, 1e10, -1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.degrees_1"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_1'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_1'], lib="jax", suffix=1)
