
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array representing a sequence of 1D conditions/choices
    condlist = np.array([[True, False, True], [False, True, False]], dtype=bool)
    choicelist = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.int32)
    default = np.array(0, dtype=np.int32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 2: 3D array representing a sequence of 2D arrays
    condlist = np.array([
        [[True, False], [False, True]],
        [[False, True], [True, False]]
    ], dtype=bool)
    choicelist = np.array([
        [[1.5, 2.5], [3.5, 4.5]],
        [[5.5, 6.5], [7.5, 8.5]]
    ], dtype=np.float32)
    default = np.array(9.9, dtype=np.float32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 3: Negative integer values
    condlist = np.array([[True, False], [False, True]], dtype=bool)
    choicelist = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    default = np.array(-9, dtype=np.int32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 4: Single condition sequence
    condlist = np.array([[True, False, True]], dtype=bool)
    choicelist = np.array([[10, 20, 30]], dtype=np.int32)
    default = np.array(-1, dtype=np.int32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 5: Float32 single condition sequence
    condlist = np.array([[True, False]], dtype=bool)
    choicelist = np.array([[10.0, 20.0]], dtype=np.float32)
    default = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 6: Large integers (int64)
    condlist = np.array([[False, True, False]], dtype=bool)
    choicelist = np.array([[100000000000, 200000000000, 300000000000]], dtype=np.int64)
    default = np.array(-1, dtype=np.int64)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 7: Float64 high precision
    condlist = np.array([[True, False]], dtype=bool)
    choicelist = np.array([[1.234567890123456, 2.345678901234567]], dtype=np.float64)
    default = np.array(0.0, dtype=np.float64)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 8: Standard int32 alternative
    condlist = np.array([[True, False]], dtype=bool)
    choicelist = np.array([[100, 200]], dtype=np.int32)
    default = np.array(0, dtype=np.int32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 9: Multiple conditions (3 conditions)
    condlist = np.array([
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ], dtype=bool)
    choicelist = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ], dtype=np.int32)
    default = np.array(9, dtype=np.int32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    # Input 10: 3D shape with 2 conditions
    condlist = np.array([[[True, False]], [[False, True]]], dtype=bool)
    choicelist = np.array([[[1.1, 2.2]], [[3.3, 4.4]]], dtype=np.float32)
    default = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({
        "condlist": copy.deepcopy(condlist),
        "choicelist": copy.deepcopy(choicelist),
        "default": copy.deepcopy(default)
    })

    return list_of_inputs

generated_inputs["jax.numpy.select_3"] = select_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.select_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.select_3'.")


check_valid('jax.numpy.select', generated_inputs['jax.numpy.select_3'], lib="jax", suffix=3)
