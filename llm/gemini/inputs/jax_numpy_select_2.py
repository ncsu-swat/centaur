
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def select_inputs():
    list_of_inputs = []

    # Input 1: 3 conditions and choices, 1D sub-arrays
    condlist = np.array(
        [
            [True, False, False, True],
            [False, True, False, False],
            [False, False, True, False],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 0.0}
        )
    )

    # Input 2: 2 conditions and choices, 2D sub-arrays
    condlist = np.array(
        [
            [[True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5]],
            [[-10.0, 20.0, -30.0], [40.0, -50.0, 60.0]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": -1.0}
        )
    )

    # Input 3: Single condition and choice
    condlist = np.array([[True, False, True, False, False]], dtype=bool)
    choicelist = np.array([[5.0, -5.0, 15.0, -15.0, 25.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 99.9}
        )
    )

    # Input 4: 4 conditions on 1D sub-arrays
    condlist = np.array(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 0.5}
        )
    )

    # Input 5: 2 conditions on 2D square matrices
    condlist = np.array(
        [
            [[True, False, False], [False, True, False], [False, False, True]],
            [[False, False, True], [False, True, False], [True, False, False]],
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": -999.0}
        )
    )

    # Input 6: Single-element sub-arrays
    condlist = np.array([[False], [True], [False]], dtype=bool)
    choicelist = np.array([[12.34], [56.78], [90.12]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 42.0}
        )
    )

    # Input 7: 4D array (2 conditions of 3D sub-arrays)
    condlist = np.zeros((2, 2, 2, 2), dtype=bool)
    condlist[0, 0, 0, 0] = True
    condlist[1, 1, 1, 1] = True
    choicelist = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.14
    choicelist[1] = -choicelist[1]
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": -0.5}
        )
    )

    # Input 8: All conditions False to evaluate default value
    condlist = np.zeros((2, 5), dtype=bool)
    choicelist = np.array(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [60.0, 70.0, 80.0, 90.0, 100.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 123.456}
        )
    )

    # Input 9: 5 conditions with float64 precision
    condlist = np.eye(5, 2, dtype=bool)
    choicelist = np.arange(10, dtype=np.float64).reshape(5, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": 0.0}
        )
    )

    # Input 10: 2 conditions with larger 1D arrays
    condlist = np.array(
        [
            [True, False] * 5,
            [False, True] * 5,
        ],
        dtype=bool,
    )
    choicelist = np.array(
        [
            np.linspace(0.0, 1.0, 10, dtype=np.float32),
            np.linspace(-1.0, 0.0, 10, dtype=np.float32),
        ]
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"condlist": condlist, "choicelist": choicelist, "default": -3.14159}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.select_2"] = select_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.select_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.select_2'.")


check_valid('jax.numpy.select', generated_inputs['jax.numpy.select_2'], lib="jax", suffix=2)
