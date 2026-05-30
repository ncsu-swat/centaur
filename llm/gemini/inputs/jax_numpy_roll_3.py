
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 1D shift and axis
    a = np.arange(10).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (2,),
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, shift and axis of length 2
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (1, 2),
        "axis": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, negative shifts
    a = np.arange(12).reshape(3, 4).astype(np.int32)
    input_dict = {
        "a": a,
        "shift": (-1, -2),
        "axis": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, 3D shift and axis
    a = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "shift": (2, -1, 3),
        "axis": (0, 1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, rolling only 2 axes
    a = np.random.randn(2, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (2, -2),
        "axis": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, rolling 3 axes with negative axes indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (1, 2, -1),
        "axis": (0, -2, -1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D boolean array
    a = (np.random.randn(5, 5) > 0).astype(bool)
    input_dict = {
        "a": a,
        "shift": (2, 2),
        "axis": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, negative axis index
    a = np.arange(5).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (-2,),
        "axis": (-1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 array, rolling 2 non-contiguous axes
    a = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "shift": (1, -1),
        "axis": (0, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "shift": (1, 1),
        "axis": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.roll_3"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_3'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_3'], lib="jax", suffix=3)
