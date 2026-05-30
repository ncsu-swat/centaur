
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logical_xor_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D boolean arrays of the same size
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, True, True, False], dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean arrays of the same size
    x = np.random.choice([True, False], size=(3, 3))
    y = np.random.choice([True, False], size=(3, 3))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D integer arrays (integers cast to boolean)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    y = np.array([0, 1, 0, 0, -1], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float arrays including negative values and zeros
    x = np.array([0.0, 1.5, -2.3, 0.0], dtype=np.float32)
    y = np.array([-1.0, 0.0, 0.0, 3.14], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting - 2D and 1D arrays
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting - row vector and column vector
    x = np.array([[True, False, True]], dtype=bool)  # shape (1, 3)
    y = np.array([[False], [True]], dtype=bool)       # shape (2, 1)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D scalar-like arrays
    x = np.array(True, dtype=bool)
    y = np.array(False, dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64 arrays of larger shapes
    x = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    y = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional arrays (4D)
    x = np.random.choice([True, False], size=(2, 3, 4, 5))
    y = np.random.choice([True, False], size=(2, 3, 4, 5))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float64 arrays with very small values
    x = np.array([[1e-15, 0.0], [0.0, -1e-15]], dtype=np.float64)
    y = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
