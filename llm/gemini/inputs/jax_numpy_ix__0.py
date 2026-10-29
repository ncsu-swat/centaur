
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ix_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive indices
    args = np.array([0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 2: 1D int64 array with negative indices
    args = np.array([-3, -1, 0, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 3: Single-element 1D int32 array
    args = np.array([5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 4: Empty 1D int32 array
    args = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 5: Single-element 1D int64 array with negative value
    args = np.array([-1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 6: 1D int32 array with non-consecutive values
    args = np.array([100, 20, 5, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 7: 1D int64 array with large range
    args = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 8: 1D int32 array with negative and positive values
    args = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 9: 1D int32 array with repeated indices
    args = np.array([0, 0, 1, 2, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 10: 1D int64 array with two elements
    args = np.array([10, 20], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 11: Empty 1D int64 array
    args = np.array([], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    # Input 12: 1D int64 array with descending values
    args = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"args": args}))

    return list_of_inputs


generated_inputs["jax.numpy.ix_"] = ix_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ix_'.")


check_valid('jax.numpy.ix_', generated_inputs['jax.numpy.ix_'], lib="jax", suffix=0)
