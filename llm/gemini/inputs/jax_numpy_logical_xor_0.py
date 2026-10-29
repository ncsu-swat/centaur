
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logical_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays, same shape
    x = np.array([True, True, False, False], dtype=np.bool_)
    y = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D integer arrays, same shape (integers evaluate truthiness)
    x = np.array([0, 1, 2, 0], dtype=np.int32)
    y = np.array([1, 0, 3, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D float arrays with negative numbers, NaNs, and zeros
    x = np.array([[-1.5, 0.0], [2.5, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 3.2], [-0.5, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting 1D to 2D
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([[False], [True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar arrays
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float64 arrays with broadcasting
    x = np.random.randn(2, 1, 4).astype(np.float64)
    y = np.random.randn(1, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 3, 3))
    y = np.random.choice([True, False], size=(2, 2, 3, 3))
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Complex number arrays
    x = np.array([1 + 1j, 0 + 0j, 0 + 2j], dtype=np.complex64)
    y = np.array([0 + 0j, 1 + 0j, -1 - 1j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Unsigned integer arrays, different shapes with broadcasting
    x = np.array([0, 255, 128], dtype=np.uint8)
    y = np.array([[0], [1]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty arrays
    x = np.array([], dtype=np.bool_)
    y = np.array([], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 5D integer arrays
    x = np.ones((1, 2, 1, 3, 2), dtype=np.int64)
    y = np.zeros((2, 1, 2, 1, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.logical_xor"] = logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_xor'.")


check_valid('jax.numpy.logical_xor', generated_inputs['jax.numpy.logical_xor'], lib="jax", suffix=0)
