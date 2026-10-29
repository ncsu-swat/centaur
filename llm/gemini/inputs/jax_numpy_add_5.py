
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def add_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with zero
    input_dict = {
        "x": 0,
        "y": np.array([1, 2, 3, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with positive integer
    input_dict = {
        "x": 5,
        "y": np.random.randn(2, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array with negative integer
    input_dict = {
        "x": -10,
        "y": np.random.randn(4, 3, 2).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array with positive integer
    input_dict = {
        "x": 42,
        "y": np.array(7, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int16 array with negative integer
    input_dict = {
        "x": -1,
        "y": np.ones((2, 2, 2, 2), dtype=np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D uint8 array with positive integer
    input_dict = {
        "x": 100,
        "y": np.array([0, 50, 150, 200], dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D boolean array with positive integer
    input_dict = {
        "x": 1,
        "y": np.array([[True, False], [False, True]], dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array with negative integer
    input_dict = {
        "x": -50,
        "y": np.random.randn(1, 10).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array with positive integer
    input_dict = {
        "x": 7,
        "y": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
            np.complex64
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D empty float32 array with integer
    input_dict = {
        "x": 0,
        "y": np.empty((0, 5), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 array with large integer
    input_dict = {
        "x": 1000000,
        "y": np.arange(-10, 10, dtype=np.int64).reshape(4, 5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.add_5"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_5'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_5'], lib="jax", suffix=5)
