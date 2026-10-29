
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def select_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D choices with int32
    condlist = np.array(
        [[True, False, False, True], [False, True, False, False]], dtype=bool
    )
    choicelist = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.int32)
    default = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 2: 2D choices with negative float32 values
    condlist = np.array(
        [
            [[True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[-1.5, -2.5, -3.5], [-4.5, -5.5, -6.5]],
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        ],
        dtype=np.float32,
    )
    default = np.array([[-99.0, -99.0, -99.0], [-99.0, -99.0, -99.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 3: Single condition and choice with float64
    condlist = np.array([[True, False, True, False, True]], dtype=bool)
    choicelist = np.array([[10.5, 20.5, 30.5, 40.5, 50.5]], dtype=np.float64)
    default = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 4: Three conditions with int64
    condlist = np.array(
        [[False, True, False], [True, False, False], [False, False, True]],
        dtype=bool,
    )
    choicelist = np.array(
        [[-1, -2, -3], [-10, -20, -30], [-100, -200, -300]], dtype=np.int64
    )
    default = np.array([-999, -999, -999], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 5: 1D choices with float32
    condlist = np.array([[True, False, False], [False, True, False]], dtype=bool)
    choicelist = np.array(
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        dtype=np.float32,
    )
    default = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 6: 3D choices with float32
    condlist = np.zeros((2, 2, 2, 2), dtype=bool)
    condlist[0, 0, 0, 0] = True
    condlist[1, 1, 1, 1] = True
    choicelist = np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0
    choicelist[1] = -5.0
    default = np.zeros((2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 7: 1D choices with int32
    condlist = np.array(
        [[True, False, True, False], [False, True, False, True]], dtype=bool
    )
    choicelist = np.array(
        [[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.int32
    )
    default = np.array([255, 255, 255, 255], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 8: Boolean choices
    condlist = np.array([[True, False, False], [False, False, True]], dtype=bool)
    choicelist = np.array([[True, True, False], [False, True, True]], dtype=bool)
    default = np.array([False, False, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 9: Four conditions with int32
    condlist = np.array(
        [[True, False], [False, True], [False, False], [False, False]], dtype=bool
    )
    choicelist = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    default = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

    # Input 10: 2D choices with int64
    condlist = np.array(
        [
            [[True, False], [False, True], [True, True]],
            [[False, True], [True, False], [False, False]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[10, 20], [30, 40], [50, 60]],
            [[-10, -20], [-30, -40], [-50, -60]],
        ],
        dtype=np.int64,
    )
    default = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": default}
        )
    )

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
