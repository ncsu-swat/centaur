
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def select_inputs():
    list_of_inputs = []

    # Input 1: 2 conditions, 1D slices, bool choices, default False
    condlist = np.array(
        [[True, False, False, True], [False, True, False, False]], dtype=bool
    )
    choicelist = np.array(
        [[True, True, False, False], [False, False, True, True]], dtype=bool
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2 conditions, 1D slices, int32 choices, default True
    condlist = np.array([[False, False, False], [False, True, False]], dtype=bool)
    choicelist = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3 conditions, 1D slices, float32 choices with negative values, default False
    condlist = np.array(
        [
            [True, False, False, False, True],
            [False, True, False, False, False],
            [False, False, True, False, False],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [-1.5, -2.5, -3.5, -4.5, -5.5],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.5, 20.5, 30.5, 40.5, 50.5],
        ],
        dtype=np.float32,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2 conditions, 2D slices (shape (2, 2, 2)), int64 choices, default True
    condlist = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[100, -200], [300, -400]],
            [[-500, 600], [-700, 800]],
        ],
        dtype=np.int64,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single condition, 1D slice, float64 choices, default False
    condlist = np.array([[True, False, True, False, True, False]], dtype=bool)
    choicelist = np.array(
        [[1.11, 2.22, 3.33, 4.44, 5.55, 6.66]], dtype=np.float64
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4 conditions, 1D slices, int16 choices, default True
    condlist = np.array(
        [
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [True, False, False],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        dtype=np.int16,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2 conditions, 2D slices (shape (2, 4, 3)), float32 choices, default False
    condlist = np.zeros((2, 4, 3), dtype=bool)
    condlist[0, :2, :] = True
    condlist[1, 2:, :] = True
    choicelist = np.linspace(-10.0, 10.0, 24, dtype=np.float32).reshape(2, 4, 3)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3 conditions, 3D slices (shape (3, 2, 2, 2)), int32 choices, default True
    condlist = np.ones((3, 2, 2, 2), dtype=bool)
    condlist[1] = False
    choicelist = np.arange(24, dtype=np.int32).reshape(3, 2, 2, 2)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2 conditions, 1D slices, int8 choices with negative values, default False
    condlist = np.array(
        [[True, True, False, False, False], [False, False, True, True, False]],
        dtype=bool,
    )
    choicelist = np.array(
        [[-10, -20, -30, -40, -50], [10, 20, 30, 40, 50]], dtype=np.int8
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3 conditions, 1D slices, float64 choices, default True
    condlist = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [0.25, 0.5, 0.75, 1.0],
            [2.0, 4.0, 6.0, 8.0],
            [-0.1, -0.2, -0.3, -0.4],
        ],
        dtype=np.float64,
    )
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2 conditions, 1D slices, boolean choices, default True
    condlist = np.array([[True, False, True], [False, True, False]], dtype=bool)
    choicelist = np.array([[False, True, False], [True, False, True]], dtype=bool)
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.select_4"] = select_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.select_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.select_4'.")


check_valid('jax.numpy.select', generated_inputs['jax.numpy.select_4'], lib="jax", suffix=4)
