
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, boolean True
    input_dict = {
        "x": True,
        "y": np.array([1, 2, 3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int64 array, boolean False
    input_dict = {
        "x": False,
        "y": np.array([[5, 6], [7, 8]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D boolean array, boolean True
    input_dict = {
        "x": True,
        "y": np.array([True, False, True, False], dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D random int32 array, boolean False
    input_dict = {
        "x": False,
        "y": np.random.randint(-50, 50, size=(2, 3, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D uint8 array, boolean True
    input_dict = {
        "x": True,
        "y": np.random.randint(0, 255, size=(5,), dtype=np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int16 array, boolean False
    input_dict = {
        "x": False,
        "y": np.array([1024, -2048, 4096], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D large int64 array with negative numbers, boolean True
    input_dict = {
        "x": True,
        "y": np.random.randint(-1000, 1000, size=(3, 4), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D small int32 array, boolean True
    input_dict = {
        "x": True,
        "y": np.array([[[1]], [[2]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D single-element boolean array, boolean False
    input_dict = {
        "x": False,
        "y": np.array([True], dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D random int32 array, boolean True
    input_dict = {
        "x": True,
        "y": np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_xor_5"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_5'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_5'], lib="jax", suffix=5)
