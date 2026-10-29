
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def glu_inputs():
    list_of_inputs = []

    # Input 1: 1D array along axis 0
    x = np.random.randn(4).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array along axis -1
    x = np.random.randn(6).astype(np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 1 (last axis)
    x = np.random.randn(3, 8).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array along axis 0 (first axis)
    x = np.random.randn(10, 5).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative values, float64
    x = -np.abs(np.random.randn(4, 6)).astype(np.float64)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array using negative axis indexing
    x = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"x": x, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array along axis -1
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array along middle axis
    x = np.random.randn(5, 8, 3).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array along axis 0, float16
    x = np.random.randn(4, 5, 6).astype(np.float16)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array along axis 3
    x = np.random.randn(2, 4, 3, 12).astype(np.float32)
    input_dict = {"x": x, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array along axis -3
    x = np.random.randn(2, 6, 5, 7).astype(np.float64)
    input_dict = {"x": x, "axis": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D array along axis 2
    x = np.random.randn(1, 2, 4, 3, 2).astype(np.float32)
    input_dict = {"x": x, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.nn.glu"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.glu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.glu'.")


check_valid('jax.nn.glu', generated_inputs['jax.nn.glu'], lib="jax", suffix=0)
