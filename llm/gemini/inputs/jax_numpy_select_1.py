
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def select_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D conditions/choices (stacked into 2D tensor)
    condlist = np.array(
        [[True, False, False, False], [False, True, False, False]], dtype=bool
    )
    choicelist = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.int32)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float choices with negative default
    condlist = np.array(
        [[False, True, False], [False, False, True]], dtype=bool
    )
    choicelist = np.array(
        [[1.5, -2.5, 3.5], [4.0, 5.0, -6.0]], dtype=np.float32
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D condition/choice arrays (stacked into 3D tensor)
    condlist = np.array(
        [[[True, False], [False, True]], [[False, True], [True, False]]],
        dtype=bool,
    )
    choicelist = np.array(
        [[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], dtype=np.int32
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3 conditions of 2D float64 arrays
    condlist = np.array(
        [
            [[True, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, True, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, True]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]],
        ],
        dtype=np.float64,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single condition and choice
    condlist = np.array([[True, False, True, False, True]], dtype=bool)
    choicelist = np.array([[100, 200, 300, 400, 500]], dtype=np.int64)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D conditions and choices (stacked into 4D tensor)
    condlist = np.zeros((2, 2, 2, 2), dtype=bool)
    condlist[0, 0, 0, 0] = True
    condlist[1, 1, 1, 1] = True
    choicelist = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4 conditions of 1D arrays
    condlist = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ],
        dtype=np.int32,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All false conditions triggering default
    condlist = np.zeros((2, 3, 3), dtype=bool)
    choicelist = np.ones((2, 3, 3), dtype=np.int32)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3 conditions of 3D float32 tensors
    condlist = np.zeros((3, 2, 2, 2), dtype=bool)
    condlist[0, :, :, 0] = True
    condlist[1, :, :, 1] = True
    choicelist = np.ones((3, 2, 2, 2), dtype=np.float32) * -3.0
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D tensor selecting between scalar values
    condlist = np.array([False, True], dtype=bool)
    choicelist = np.array([10, 20], dtype=np.int32)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.select_1"] = select_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.select_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.select_1'.")


check_valid('jax.numpy.select', generated_inputs['jax.numpy.select_1'], lib="jax", suffix=1)
