
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D conditions stacked as a 2D array
    condlist = np.array([[True, False, False], [False, True, False]])
    choicelist = np.array([[1, 2, 3], [10, 20, 30]])
    default = 0
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D conditions stacked, with float values and negative default
    condlist = np.array([[False, True], [True, False]])
    choicelist = np.array([[-1.5, -2.5], [10.5, 20.5]])
    default = -1
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays stacked
    condlist = np.array([
        [[True, False, True], [False, True, False], [True, False, True]],
        [[False, True, False], [True, False, True], [False, True, False]]
    ])
    choicelist = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    ])
    default = 99
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays stacked
    condlist = np.array([
        np.ones((2, 2, 2), dtype=bool),
        np.zeros((2, 2, 2), dtype=bool)
    ])
    choicelist = np.array([
        np.full((2, 2, 2), 5),
        np.full((2, 2, 2), 15)
    ])
    default = -99
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcastable dimensions
    condlist = np.array([[[True], [False], [True]], [[False], [True], [False]]])
    choicelist = np.array([np.ones((3, 3)) * 1.0, np.ones((3, 3)) * 2.0])
    default = 0
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar / 0D arrays (represented as 1D array of options)
    condlist = np.array([True, False])
    choicelist = np.array([42, 84])
    default = -1
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 types
    condlist = np.array([[True, False, True], [False, True, False]])
    choicelist = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
    default = 100
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 4D arrays stacked
    condlist = np.random.choice([True, False], size=(3, 2, 2, 2))
    choicelist = np.random.randint(-10, 10, size=(3, 2, 2, 2))
    default = 5
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays
    condlist = np.random.choice([True, False], size=(5, 100))
    choicelist = np.random.randint(-100, 100, size=(5, 100))
    default = 42
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Diagonal matrices
    condlist = np.array([np.eye(4, dtype=bool), np.eye(4, dtype=bool)[::-1]])
    choicelist = np.array([np.ones((4, 4)) * 10, np.ones((4, 4)) * 20])
    default = -5
    input_dict = {"condlist": condlist, "choicelist": choicelist, "default": default}
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
