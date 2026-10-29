
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logical_and_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays of same shape
    x = np.array([True, True, False, False], dtype=np.bool_)
    y = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D boolean arrays
    x = np.array([[True, False], [True, True]], dtype=np.bool_)
    y = np.array([[False, False], [True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D integer arrays with positive, negative, and zero values
    x = np.array([0, 1, -1, 5, 0], dtype=np.int32)
    y = np.array([1, 0, -2, 0, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting 2D and 1D float32 arrays
    x = np.array([[0.0], [1.5], [-2.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar arrays
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D arrays with shape (2, 3, 4)
    x = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Broadcasting 0D array with 2D array
    x = np.array([[1, 0], [2, 3]], dtype=np.int64)
    y = np.array(1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Float64 arrays including inf and nan
    x = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float64)
    y = np.array([1.0, 0.0, 2.0, 0.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D broadcasting arrays
    x = np.random.choice([True, False], size=(1, 2, 1, 4)).astype(np.bool_)
    y = np.random.choice([True, False], size=(3, 1, 5, 4)).astype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty arrays
    x = np.empty((0, 3), dtype=np.bool_)
    y = np.empty((0, 3), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Complex numbers
    x = np.array([0 + 0j, 1 + 0j, 0 + 1j, -1 - 1j], dtype=np.complex64)
    y = np.array([1 + 1j, 0 + 0j, 1 + 0j, 0 + 0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.logical_and"] = logical_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_and'.")


check_valid('jax.numpy.logical_and', generated_inputs['jax.numpy.logical_and'], lib="jax", suffix=0)
