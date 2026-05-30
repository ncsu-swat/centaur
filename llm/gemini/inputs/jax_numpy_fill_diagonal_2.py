
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 3x3 square matrix
    a = np.zeros((3, 3), dtype=np.int32)
    val = 1
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 5x5 square matrix with negative fill
    a = np.ones((5, 5), dtype=np.float32)
    val = -5
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x5 non-square matrix
    a = np.zeros((3, 5), dtype=np.int32)
    val = 2
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 5x3 non-square matrix
    a = np.zeros((5, 3), dtype=np.float32)
    val = 4
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D square array (2x2x2)
    a = np.zeros((2, 2, 2), dtype=np.float32)
    val = 9
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D square array (3x3x3x3)
    a = np.zeros((3, 3, 3, 3), dtype=np.int32)
    val = -1
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 10x10 float32 matrix
    a = np.random.randn(10, 10).astype(np.float32)
    val = 100
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 8x8 int32 matrix
    a = np.random.randint(-10, 10, size=(8, 8)).astype(np.int32)
    val = 7
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D square array (4x4x4) float64
    a = np.ones((4, 4, 4), dtype=np.float64)
    val = 42
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6x6 uint8 matrix
    a = np.zeros((6, 6), dtype=np.uint8)
    val = 0
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_2"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_2'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_2'], lib="jax", suffix=2)
