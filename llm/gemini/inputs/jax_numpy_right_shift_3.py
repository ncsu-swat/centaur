
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def right_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D array of int32 shifts
    input_dict = {
        "x1": 16,
        "x2": np.array([1, 2, 3, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of int64 shifts with zero shift
    input_dict = {
        "x1": 0,
        "x2": np.array([0, 1, 2, 5], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of int32 shifts
    input_dict = {
        "x1": 1024,
        "x2": np.array([[1, 2], [3, 4]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer input with int32 shifts
    input_dict = {
        "x1": -64,
        "x2": np.array([1, 2, 3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of int64 shifts
    input_dict = {
        "x1": 255,
        "x2": np.zeros((3, 2, 2), dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar array shift
    input_dict = {
        "x1": 1,
        "x2": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large integer with 1D int64 shifts
    input_dict = {
        "x1": 1000000,
        "x2": np.array([2, 4, 6, 8, 10], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array of int32 shifts
    input_dict = {
        "x1": 42,
        "x2": np.arange(1, 9, dtype=np.int32).reshape((2, 4)),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative one with int32 shifts
    input_dict = {
        "x1": -1,
        "x2": np.array([0, 1, 2, 3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of int32 shifts
    input_dict = {
        "x1": 128,
        "x2": np.ones((2, 1, 3, 1), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int64 shifts
    input_dict = {
        "x1": 7,
        "x2": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    }
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
