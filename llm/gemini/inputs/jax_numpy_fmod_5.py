
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {
        "x1": 5,
        "x2": np.array([2.0, 3.0, 4.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative integers
    input_dict = {
        "x1": -7,
        "x2": np.array([[2, -3], [4, -5]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero dividend, 1D int64 array
    input_dict = {
        "x1": 0,
        "x2": np.array([1, 2, 3], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar tensor float64
    input_dict = {
        "x1": 10,
        "x2": np.array(3.5, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array with positive and negative floats
    input_dict = {
        "x1": -15,
        "x2": np.array([[[2.5, -4.0], [3.0, 5.5]]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array
    input_dict = {
        "x1": 100,
        "x2": np.full((2, 2, 2, 2), 7.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array with all negative values
    input_dict = {
        "x1": 1,
        "x2": np.array([-2, -3, -4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer dividend, 1D int64 array
    input_dict = {
        "x1": -8,
        "x2": np.array([3, 5, 7, 9], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 array with fractional values
    input_dict = {
        "x1": 42,
        "x2": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array
    input_dict = {
        "x1": -100,
        "x2": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D int32 array with shape (1, 1, 1, 1)
    input_dict = {
        "x1": 17,
        "x2": np.array([[[[3]]]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fmod_5"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_5'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_5'], lib="jax", suffix=5)
