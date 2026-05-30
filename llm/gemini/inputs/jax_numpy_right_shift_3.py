
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def right_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D array of int32, shift by small values
    x1 = 16
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of int32, larger x1
    x1 = 1024
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x1 is zero, 1D array of int32
    x1 = 0
    x2 = np.array([0, 1, 2, 3], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of int32
    x1 = 255
    x2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of ones with int64 dtype
    x1 = 12345
    x2 = np.ones((3, 3), dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with single element, int32 dtype
    x1 = 64
    x2 = np.array([3], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D random integer array
    x1 = 4096
    x2 = np.random.randint(0, 10, size=(5, 5), dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of int32 dtype
    x1 = 8
    x2 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer x1, 2D array of int32
    x1 = 1048576
    x2 = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array of zeros
    x1 = 1
    x2 = np.zeros((2, 2, 2), dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.right_shift_3"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_3'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_3'], lib="jax", suffix=3)
