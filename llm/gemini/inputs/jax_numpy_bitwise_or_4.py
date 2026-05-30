
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 array and integer
    x = 5
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: Negative integer and 2D int16 array
    x = -1
    y = np.array([[0, 1], [2, 3]], dtype=np.int16)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: Zero and 2D random int32 array
    x = 0
    y = np.random.randint(-100, 100, size=(5, 5), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Large integer and 3D int64 array
    x = 1024
    y = np.array([[[12], [24]]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: Boolean array and integer 1
    x = 1
    y = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: Negative integer and uint8 3D array
    x = -50
    y = np.random.randint(0, 255, size=(2, 3, 4), dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: Hex mask integer and 1D int32 array
    x = 0xFFFF
    y = np.array([256, 512, 1024], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: Very large integer and int64 array
    x = 123456789
    y = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: Negative integer and int8 1D array
    x = -2
    y = np.random.randint(-10, 10, size=(10,), dtype=np.int8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: Integer with 3D broadcastable array
    x = 7
    y = np.random.randint(0, 10, size=(3, 1, 3), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 11: Zero and boolean array
    x = 0
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

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
