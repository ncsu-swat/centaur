
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, y is True
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int16 array with negative values, y is False
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int16)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, y is True
    x = np.arange(8, dtype=np.int32).reshape((2, 2, 2))
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D int64 array, y is False
    x = np.array(-42, dtype=np.int64)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D bool array, y is True
    x = np.zeros((2, 2, 2, 2), dtype=bool)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int8 array, y is True
    x = np.array([[127, -128], [0, 1]], dtype=np.int8)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array, y is False
    x = np.array([1000000, 2000000], dtype=np.int32)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D int32 array, y is True
    x = np.ones((2, 1, 2, 1, 2), dtype=np.int32)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D bool array, y is False
    x = np.array([[True, False], [False, True]], dtype=bool)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64 array with mixed signs, y is True
    x = np.array([[[-10, 20], [30, -40]], [[50, -60], [-70, 80]]], dtype=np.int64)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_xor_4"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_4'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_4'], lib="jax", suffix=4)
