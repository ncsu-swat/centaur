
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single split point
    ary = np.arange(10, dtype=np.int32)
    input_dict = {"ary": ary, "indices_or_sections": (5,), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, multiple split points
    ary = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    input_dict = {"ary": ary, "indices_or_sections": (3, 7), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, split along axis 0
    ary = np.ones((6, 4), dtype=np.float64)
    input_dict = {"ary": ary, "indices_or_sections": (2, 4), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, split along axis 1
    ary = np.arange(24, dtype=np.int64).reshape(4, 6)
    input_dict = {"ary": ary, "indices_or_sections": (1, 3, 5), "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative axis
    ary = np.zeros((5, 8), dtype=np.bool_)
    input_dict = {"ary": ary, "indices_or_sections": (2, 6), "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, split along axis 2
    ary = np.random.randn(3, 4, 8).astype(np.float32)
    input_dict = {"ary": ary, "indices_or_sections": (2, 5), "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, negative axis -2
    ary = np.random.randn(2, 9, 3).astype(np.complex64)
    input_dict = {"ary": ary, "indices_or_sections": (3, 6), "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tuple of indices (returns array as single chunk)
    ary = np.arange(8, dtype=np.uint8)
    input_dict = {"ary": ary, "indices_or_sections": (), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, split along axis 1
    ary = np.arange(48, dtype=np.int16).reshape(2, 6, 2, 2)
    input_dict = {"ary": ary, "indices_or_sections": (1, 4), "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, multiple split points within range
    ary = np.arange(10, dtype=np.float32)
    input_dict = {"ary": ary, "indices_or_sections": (2, 4, 7), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array with axis 0
    ary = np.ones((7, 3, 2), dtype=np.int8)
    input_dict = {"ary": ary, "indices_or_sections": (1, 2, 4), "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.array_split_3"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_3'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_3'], lib="jax", suffix=3)
