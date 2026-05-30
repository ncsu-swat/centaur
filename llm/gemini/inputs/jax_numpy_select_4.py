
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays represented as 2D array, integer choices, default True
    condlist = np.array([
        [True, False, True, False, True],
        [False, True, False, True, False]
    ], dtype=bool)
    choicelist = np.array([
        [10, 20, 30, 40, 50],
        [11, 21, 31, 41, 51]
    ], dtype=np.int32)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays represented as 3D array, float32 choices, default False
    condlist = np.array([
        [[True, False], [False, True]],
        [[False, True], [True, False]]
    ], dtype=bool)
    choicelist = np.array([
        [[1.5, 2.5], [3.5, 4.5]],
        [[5.5, 6.5], [7.5, 8.5]]
    ], dtype=np.float32)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays represented as 3D array, int64 choices, default True
    condlist = np.array([
        np.ones((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=bool)
    ])
    choicelist = np.array([
        np.arange(4, dtype=np.int64).reshape((2, 2)),
        np.arange(4, 8, dtype=np.int64).reshape((2, 2))
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays represented as 2D array, float64 choices, default False
    condlist = np.array([
        [False, False, False],
        [False, False, True]
    ], dtype=bool)
    choicelist = np.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6]
    ], dtype=np.float64)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D random arrays, float32 choices, default True
    condlist = np.array([
        np.random.choice([True, False], size=(3, 2)).astype(bool),
        np.random.choice([True, False], size=(3, 2)).astype(bool)
    ])
    choicelist = np.array([
        np.random.randn(3, 2).astype(np.float32),
        np.random.randn(3, 2).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element arrays, int32 choices, default False
    condlist = np.array([
        [True],
        [False]
    ], dtype=bool)
    choicelist = np.array([
        [42],
        [24]
    ], dtype=np.int32)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D arrays with negative values, default True
    condlist = np.array([
        [[True, True], [False, False]],
        [[False, False], [True, True]]
    ], dtype=bool)
    choicelist = np.array([
        [[-1.0, -2.0], [-3.0, -4.0]],
        [[-5.0, -6.0], [-7.0, -8.0]]
    ], dtype=np.float32)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D arrays (length 50), int32 choices, default False
    condlist = np.array([
        np.random.choice([True, False], size=(50,)).astype(bool),
        np.random.choice([True, False], size=(50,)).astype(bool)
    ])
    choicelist = np.array([
        np.random.randint(-100, 100, size=(50,)).astype(np.int32),
        np.random.randint(-100, 100, size=(50,)).astype(np.int32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D random arrays, float32 choices, default True
    condlist = np.array([
        np.random.choice([True, False], size=(4, 3)).astype(bool),
        np.random.choice([True, False], size=(4, 3)).astype(bool)
    ])
    choicelist = np.array([
        np.random.randn(4, 3).astype(np.float32),
        np.random.randn(4, 3).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, boolean choices, default False
    condlist = np.array([
        [[True, False], [False, True]],
        [[False, True], [True, False]]
    ], dtype=bool)
    choicelist = np.array([
        [[True, True], [False, False]],
        [[False, False], [True, True]]
    ], dtype=bool)
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": False
    }
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
