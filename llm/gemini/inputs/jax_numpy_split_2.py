
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array split at two indices
    ary = np.arange(10, dtype=np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [3, 7],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array split along axis 0
    ary = np.random.randn(6, 4).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [2, 4],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array split along axis 1
    ary = np.random.randn(3, 8).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [2, 5],
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array split along axis 2
    ary = np.random.randint(0, 10, size=(2, 3, 10)).astype(np.int32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [4, 8],
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array split along axis 1
    ary = np.random.randn(2, 5, 2, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [1, 3],
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with negative axis
    ary = np.random.randn(4, 6).astype(np.float64)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [2],
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty indices list
    ary = np.random.randn(5).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array split along axis 0
    ary = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [5],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array split along axis 0
    ary = np.random.randn(9, 3, 3).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [3, 6],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean 1D array
    ary = np.array([True, False, True, True, False, False, True, False])
    input_dict = {
        "ary": ary,
        "indices_or_sections": [4],
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array split into multiple parts
    ary = np.random.randn(4, 12).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": [3, 6, 9],
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.split_2"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_2'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_2'], lib="jax", suffix=2)
