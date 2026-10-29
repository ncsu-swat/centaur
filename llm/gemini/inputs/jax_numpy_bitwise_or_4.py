
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 array
    input_dict = {"x": 1, "y": np.array([0, 1, 2, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int64 array with zero
    input_dict = {
        "x": 0,
        "y": np.array([[10, 20], [30, 40]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer and negative int32 array
    input_dict = {"x": -1, "y": np.array([-5, 0, 5], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array
    input_dict = {
        "x": 7,
        "y": np.random.randint(0, 100, size=(2, 3, 4), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty 1D int32 array
    input_dict = {"x": 2, "y": np.array([], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar int32 array
    input_dict = {"x": 15, "y": np.array(5, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int64 array
    input_dict = {
        "x": 3,
        "y": np.array([[[1, -2]], [[3, -4]]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array
    input_dict = {
        "x": 255,
        "y": np.random.randint(0, 255, size=(4, 4), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int64 array with negative scalar
    input_dict = {
        "x": -10,
        "y": np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D column vector with int32
    input_dict = {
        "x": 42,
        "y": np.random.randint(0, 1000, size=(5, 1), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int64 array
    input_dict = {
        "x": 1024,
        "y": np.array([100, 200, 300, 400], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_or_4"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_4'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_4'], lib="jax", suffix=4)
