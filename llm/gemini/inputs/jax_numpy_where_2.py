
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def where_2_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, size padded with 0
    condition = np.array([True, False, True, False, True], dtype=np.bool_)
    input_dict = {"condition": condition, "size": 6, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer array, negative fill_value
    condition = np.array([0, 1, 2, 0, -1, 0], dtype=np.int32)
    input_dict = {"condition": condition, "size": 10, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array, exact size
    condition = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"condition": condition, "size": 2, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array, truncated size
    condition = np.array([[1.5, 0.0, -2.3], [0.0, 4.1, 0.0]], dtype=np.float32)
    input_dict = {"condition": condition, "size": 2, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D boolean array
    condition = np.zeros((2, 3, 4), dtype=np.bool_)
    condition[0, 1, 2] = True
    condition[1, 2, 3] = True
    input_dict = {"condition": condition, "size": 5, "fill_value": 99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D all-true condition, larger size
    condition = np.ones(8, dtype=np.bool_)
    input_dict = {"condition": condition, "size": 16, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D all-false condition
    condition = np.zeros(10, dtype=np.bool_)
    input_dict = {"condition": condition, "size": 4, "fill_value": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D integer array (int64) with size 0
    condition = np.array([[0, 0], [1, 0]], dtype=np.int64)
    input_dict = {"condition": condition, "size": 0, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D boolean tensor
    condition = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(
        np.bool_
    )
    input_dict = {"condition": condition, "size": 16, "fill_value": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 condition with non-zero elements
    condition = np.array([0.0, 1e-5, -3.2, 0.0, 4.0], dtype=np.float64)
    input_dict = {"condition": condition, "size": 8, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int16 condition
    condition = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    input_dict = {"condition": condition, "size": 25, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.where_2"] = where_2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.where_2'.")


check_valid('jax.numpy.where', generated_inputs['jax.numpy.where_2'], lib="jax", suffix=2)
