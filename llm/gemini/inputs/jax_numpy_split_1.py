
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array split into 2 equal sections
    ary = np.random.randn(6).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array split along axis 0 into 4 equal sections
    ary = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 4,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array split along axis 1 into 2 equal sections
    ary = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array split along axis 0 into 3 equal sections
    ary = np.random.randn(6, 3, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array split along axis 1 into 4 equal sections
    ary = np.random.randn(2, 8, 4).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 4,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array split along axis 3 into 5 equal sections
    ary = np.random.randn(2, 3, 4, 10).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 5,
        "axis": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array split along negative axis -1 (axis 1) into 3 equal sections
    ary = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D integer array split into 3 equal sections
    ary = np.arange(12, dtype=np.int32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D double precision array split along axis 0 into 3 equal sections
    ary = np.random.randn(9, 3, 3).astype(np.float64)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 3,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array split along axis 4 into 2 equal sections
    ary = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D boolean array split along axis 0 into 2 equal sections
    ary = np.random.choice([True, False], size=(8, 2))
    input_dict = {
        "ary": ary,
        "indices_or_sections": 2,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3D array split along negative axis -1 (axis 2) into 4 equal sections
    ary = np.random.randn(5, 5, 12).astype(np.float32)
    input_dict = {
        "ary": ary,
        "indices_or_sections": 4,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.split_1"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_1'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_1'], lib="jax", suffix=1)
