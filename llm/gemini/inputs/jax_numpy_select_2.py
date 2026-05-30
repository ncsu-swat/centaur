
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    condlist = np.stack([
        np.array([True, False, True, False], dtype=bool),
        np.array([False, True, False, False], dtype=bool)
    ])
    choicelist = np.stack([
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays
    condlist = np.stack([
        np.random.choice([True, False], size=(3, 3)),
        np.random.choice([True, False], size=(3, 3)),
        np.random.choice([True, False], size=(3, 3))
    ])
    choicelist = np.stack([
        np.random.randn(3, 3).astype(np.float32),
        np.random.randn(3, 3).astype(np.float32),
        np.random.randn(3, 3).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays
    condlist = np.stack([
        np.random.choice([True, False], size=(2, 2, 2)),
        np.random.choice([True, False], size=(2, 2, 2))
    ])
    choicelist = np.stack([
        np.random.randn(2, 2, 2).astype(np.float32),
        np.random.randn(2, 2, 2).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars)
    condlist = np.stack([
        np.array(True, dtype=bool),
        np.array(False, dtype=bool)
    ])
    choicelist = np.stack([
        np.array(5.5, dtype=np.float32),
        np.array(-5.5, dtype=np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcast shapes
    condlist = np.stack([
        np.array([[True], [False], [True]], dtype=bool),
        np.array([[False], [True], [False]], dtype=bool)
    ])
    choicelist = np.stack([
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[10.0, 20.0]], dtype=np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 42.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 types
    condlist = np.stack([
        np.random.choice([True, False], size=(4,)),
        np.random.choice([True, False], size=(4,))
    ])
    choicelist = np.stack([
        np.random.randn(4).astype(np.float64),
        np.random.randn(4).astype(np.float64)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Only 1 item in lists
    condlist = np.stack([
        np.array([True, False, True], dtype=bool)
    ])
    choicelist = np.stack([
        np.array([3.14, 2.71, 1.41], dtype=np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 2D arrays
    condlist = np.stack([
        np.random.choice([True, False], size=(10, 10)),
        np.random.choice([True, False], size=(10, 10)),
        np.random.choice([True, False], size=(10, 10))
    ])
    choicelist = np.stack([
        np.random.randn(10, 10).astype(np.float32),
        np.random.randn(10, 10).astype(np.float32),
        np.random.randn(10, 10).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": 9.99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays
    condlist = np.stack([
        np.random.choice([True, False], size=(2, 2, 2, 2)),
        np.random.choice([True, False], size=(2, 2, 2, 2))
    ])
    choicelist = np.stack([
        np.random.randn(2, 2, 2, 2).astype(np.float32),
        np.random.randn(2, 2, 2, 2).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed broadcast dimensions
    condlist = np.stack([
        np.random.choice([True, False], size=(5, 1)),
        np.random.choice([True, False], size=(5, 1))
    ])
    choicelist = np.stack([
        np.random.randn(5, 5).astype(np.float32),
        np.random.randn(5, 5).astype(np.float32)
    ])
    input_dict = {
        "condlist": condlist,
        "choicelist": choicelist,
        "default": -100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
