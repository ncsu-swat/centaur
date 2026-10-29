
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with negative and positive values
    x = np.array([-5, 2, 0, -5, 3, 2, 8, -1], dtype=np.int32)
    input_dict = {"x": x, "size": 10, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int64 array
    x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int64)
    input_dict = {"x": x, "size": 12, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"x": x, "size": 5, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array
    x = np.array(
        [[[1.0, 2.0], [3.0, 1.0]], [[2.0, 4.0], [5.0, 3.0]]], dtype=np.float32
    )
    input_dict = {"x": x, "size": 10, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array with negative and positive values
    x = np.array([-1.5, 0.0, 2.5, -1.5, 3.7, 0.0], dtype=np.float64)
    input_dict = {"x": x, "size": 8, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array
    x = np.array(
        [[True, False], [False, False], [True, True]], dtype=np.bool_
    )
    input_dict = {"x": x, "size": 4, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int16 array with duplicate elements
    x = np.array([10, 20, 10, 30, 20, 40], dtype=np.int16)
    input_dict = {"x": x, "size": 6, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int32 array
    x = np.zeros((2, 1, 3, 2), dtype=np.int32)
    x[0, 0, 1, 1] = 5
    x[1, 0, 2, 0] = -3
    input_dict = {"x": x, "size": 15, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with all identical elements
    x = np.full((3, 4), 7.0, dtype=np.float32)
    input_dict = {"x": x, "size": 5, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D uint8 array
    x = np.array([0, 128, 255, 128, 0, 64], dtype=np.uint8)
    input_dict = {"x": x, "size": 10, "fill_value": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int64 array
    x = np.array(
        [[[-2, 3], [3, -2]], [[0, 1], [-1, 0]], [[5, 5], [-3, 2]]],
        dtype=np.int64,
    )
    input_dict = {"x": x, "size": 20, "fill_value": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unique_inverse_4"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_4'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_4'], lib="jax", suffix=4)
