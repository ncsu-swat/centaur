
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def invert_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D int32 array
    x = np.arange(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D int32 array with positive and negative integers
    x = np.array([[-5, 0, 5], [10, -20, 30]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar int64 array
    x = np.array(42, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D int64 array
    x = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D int32 array
    x = np.ones((2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 1D array of int32
    x = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 3D boolean array
    x = np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D int64 array
    x = np.array([[0, -1], [1, -2]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.invert"] = invert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.invert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.invert'.")


check_valid('jax.numpy.invert', generated_inputs['jax.numpy.invert'], lib="jax", suffix=0)
