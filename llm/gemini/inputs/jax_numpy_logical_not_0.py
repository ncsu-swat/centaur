
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logical_not_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    x = np.array([True, False, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D integer array with negative, zero, and positive values
    x = np.array([-5, 0, 1, 10, -1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float32 array with zeros and non-zeros
    x = np.array([[0.0, 1.5], [-2.3, 0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar boolean array
    x = np.array(True, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D scalar float array with zero
    x = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D int8 array
    x = np.zeros((2, 3, 4), dtype=np.int8)
    x[0, 1, 2] = 1
    x[1, 2, 3] = -1
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D float64 array with inf and nan
    x = np.array([[np.nan, 0.0], [np.inf, -np.inf]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D boolean array
    x = np.ones((2, 2, 2, 2), dtype=np.bool_)
    x[0, 0, 0, 0] = False
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D complex array
    x = np.array([0 + 0j, 1 + 0j, 0 + 2j, -1 - 1j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D uint8 array
    x = np.array([[0, 255], [128, 0]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: 1D int64 array with large values
    x = np.array([0, 2**60, -(2**60)], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_not'.")


check_valid('jax.numpy.logical_not', generated_inputs['jax.numpy.logical_not'], lib="jax", suffix=0)
