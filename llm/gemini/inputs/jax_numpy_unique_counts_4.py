
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"x": x, "size": 4, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array
    x = np.array([[True, True], [False, True]], dtype=np.bool_)
    input_dict = {"x": x, "size": 5, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array with positive and negative values
    x = np.array([-5, 3, -5, 0, 10, 3], dtype=np.int32)
    input_dict = {"x": x, "size": 6, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32 array
    x = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype=np.int32)
    input_dict = {"x": x, "size": 10, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array
    x = np.arange(-12, 12, dtype=np.int64).reshape((2, 3, 4))
    input_dict = {"x": x, "size": 30, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32 array
    x = np.array([-1.5, 0.0, 2.5, -1.5, 3.0], dtype=np.float32)
    input_dict = {"x": x, "size": 8, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array
    x = np.array(
        [[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
        dtype=np.float64,
    )
    input_dict = {"x": x, "size": 12, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D uint8 array
    x = np.array([255, 0, 128, 255, 0], dtype=np.uint8)
    input_dict = {"x": x, "size": 5, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int16 array
    x = np.zeros((2, 2, 2, 2), dtype=np.int16)
    x[0, 1, 0, 1] = 5
    x[1, 0, 1, 0] = -3
    input_dict = {"x": x, "size": 6, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with size exactly matching number of unique elements
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict = {"x": x, "size": 3, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float32 array
    x = np.ones((2, 3, 2), dtype=np.float32)
    input_dict = {"x": x, "size": 4, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unique_counts_4"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_4'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_4'], lib="jax", suffix=4)
