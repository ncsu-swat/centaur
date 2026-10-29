
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argwhere_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, size greater than nonzeros
    a = np.array([0, 1, 0, 3, 4], dtype=np.int32)
    input_dict = {"a": a, "size": 5, "fill_value": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array, negative and positive values, exact size
    a = np.array([[-1.5, 0.0], [2.3, 0.0]], dtype=np.float32)
    input_dict = {"a": a, "size": 2, "fill_value": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, size smaller than nonzeros (truncation)
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"a": a, "size": 2, "fill_value": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array
    a = np.array([True, False, True, False, False], dtype=np.bool_)
    input_dict = {"a": a, "size": 4, "fill_value": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float array with zeros and nonzeros
    a = np.zeros((2, 3, 4), dtype=np.float64)
    a[0, 1, 2] = 5.0
    a[1, 2, 3] = -3.2
    input_dict = {"a": a, "size": 4, "fill_value": 99.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All-zero 2D array
    a = np.zeros((3, 3), dtype=np.int32)
    input_dict = {"a": a, "size": 3, "fill_value": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, size=0
    a = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"a": a, "size": 0, "fill_value": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array with float values
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    a[0, 0, 0, 1] = 1.0
    a[1, 1, 0, 0] = -2.0
    input_dict = {"a": a, "size": 3, "fill_value": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with large fill_value
    a = np.array([[0, 0, 1], [0, 2, 0], [3, 0, 0]], dtype=np.int16)
    input_dict = {"a": a, "size": 6, "fill_value": 1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float array with negative fill_value
    a = np.array([0.0, -0.5, 0.0, 1.2, -3.4], dtype=np.float64)
    input_dict = {"a": a, "size": 6, "fill_value": -999.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argwhere_3"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_3'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_3'], lib="jax", suffix=3)
