
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, exact matching 1D array
    a = np.zeros((3, 3), dtype=np.float32)
    val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square int32 matrix, negative integers
    a = np.ones((4, 4), dtype=np.int32)
    val = np.array([-1, -5, -10, -20], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular matrix (rows < cols), float64
    a = np.zeros((3, 5), dtype=np.float64)
    val = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D rectangular matrix (rows > cols), int64
    a = np.zeros((6, 2), dtype=np.int64)
    val = np.array([42, 99], dtype=np.int64)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D cube (equal dimensions), float32
    a = np.zeros((3, 3, 3), dtype=np.float32)
    val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D hypercube (equal dimensions), float64
    a = np.zeros((2, 2, 2, 2), dtype=np.float64)
    val = np.array([3.14, 2.71], dtype=np.float64)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D matrix with val shorter than diagonal (repeated)
    a = np.zeros((5, 5), dtype=np.float32)
    val = np.array([7.0, 8.0], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D matrix with val longer than diagonal (truncated)
    a = np.zeros((2, 2), dtype=np.int32)
    val = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array
    a = np.zeros((3, 3), dtype=bool)
    val = np.array([True, False, True], dtype=bool)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D hypercube (equal dimensions), int32
    a = np.zeros((2, 2, 2, 2, 2), dtype=np.int32)
    val = np.array([-1, 1], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single-element array broadcast across diagonal
    a = np.zeros((4, 4), dtype=np.float32)
    val = np.array([-99.5], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fill_diagonal_4"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_4'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_4'], lib="jax", suffix=4)
