
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32
    a = np.zeros((3, 3), dtype=np.float32)
    val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix, int32, 1-element val
    a = np.zeros((4, 4), dtype=np.int32)
    val = np.array([5], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D non-square (tall) matrix, float32
    a = np.zeros((5, 3), dtype=np.float32)
    val = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D non-square (wide) matrix, int32, too many elements in val
    a = np.zeros((3, 5), dtype=np.int32)
    val = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D square matrix, float64, too few elements in val (repeated)
    a = np.zeros((6, 6), dtype=np.float64)
    val = np.array([1.1, 2.2], dtype=np.float64)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D square tensor, float32
    a = np.zeros((2, 2, 2), dtype=np.float32)
    val = np.array([9.9], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D square tensor, int32
    a = np.zeros((3, 3, 3, 3), dtype=np.int32)
    val = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D matrix, negative float32 values
    a = np.full((3, 3), -1.0, dtype=np.float32)
    val = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D matrix, int64
    a = np.zeros((5, 5), dtype=np.int64)
    val = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D square tensor, float32
    a = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    val = np.array([0.5, 1.5], dtype=np.float32)
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_1"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_1'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_1'], lib="jax", suffix=1)
