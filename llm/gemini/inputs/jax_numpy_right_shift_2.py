
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def right_shift_inputs():
    list_of_inputs = []

    # Input 1: uint8 1D array, shift by 1
    x1 = np.array([2, 4, 8, 16], dtype=np.uint8)
    x2 = 1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: uint8 2D array, shift by 2
    x1 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: uint8 3D array, shift by 3
    x1 = np.array([[[100, 200], [150, 250]]], dtype=np.uint8)
    x2 = 3
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: uint8 1D array, shift by 0
    x1 = np.array([128, 255], dtype=np.uint8)
    x2 = 0
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: uint8 1D array, shift by 4
    x1 = np.array([240, 128, 64], dtype=np.uint8)
    x2 = 4
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: uint8 4D array, shift by 5
    x1 = np.ones((2, 2, 2, 2), dtype=np.uint8) * 128
    x2 = 5
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: uint8 1D array, shift by 7
    x1 = np.array([128, 255], dtype=np.uint8)
    x2 = 7
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: uint8 5D array, shift by 1
    x1 = np.arange(32, dtype=np.uint8).reshape((2, 2, 2, 2, 2))
    x2 = 1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: uint8 0-D array, shift by 1
    x1 = np.array(16, dtype=np.uint8)
    x2 = 1
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: uint8 3D random array, shift by 2
    x1 = np.random.randint(0, 256, size=(3, 3, 3)).astype(np.uint8)
    x2 = 2
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.right_shift_2"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_2'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_2'], lib="jax", suffix=2)
