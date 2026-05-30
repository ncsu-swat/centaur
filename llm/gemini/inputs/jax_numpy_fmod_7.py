
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar integers
    input_dict = {
        "x1": int(7),
        "x2": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative scalar integer and positive divisor
    input_dict = {
        "x1": np.int32(-15),
        "x2": np.int32(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays of int32
    input_dict = {
        "x1": np.array([10, -11, 12, -13], dtype=np.int32),
        "x2": np.array([3, 3, 5, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays of int64
    input_dict = {
        "x1": np.array([[5, 12], [17, 23]], dtype=np.int64),
        "x2": np.array([[2, 5], [3, 7]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcast compatibility (2D and 1D)
    input_dict = {
        "x1": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        "x2": np.array([3, 7, 11], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcast scalar divisor to 1D array
    input_dict = {
        "x1": np.array([100, 200, 300], dtype=np.int16),
        "x2": np.int16(35)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays of int32
    input_dict = {
        "x1": np.ones((2, 3, 2), dtype=np.int32) * 25,
        "x2": np.array([[[3, 4], [5, 6], [7, 8]], [[2, 3], [4, 5], [6, 7]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative divisor with int64 arrays
    input_dict = {
        "x1": np.array([-1000, 2000, -3000], dtype=np.int64),
        "x2": np.array([-150, -150, -150], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero values in x1 (numerator)
    input_dict = {
        "x1": np.array([0, 0, 0], dtype=np.int32),
        "x2": np.array([5, 10, 15], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int8 datatype inputs with broadcast
    input_dict = {
        "x1": np.array([[[10], [20]]], dtype=np.int8),
        "x2": np.array([[[3, 4]]], dtype=np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmod_7"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_7'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_7'], lib="jax", suffix=7)
