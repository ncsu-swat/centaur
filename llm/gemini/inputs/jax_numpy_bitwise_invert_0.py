
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_invert_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive and negative integers
    x = np.array([-10, -1, 0, 1, 10, 42], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D int64 array
    x = np.array([[-100, 200], [0, -1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar int32 array
    x = np.array(-5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D boolean array
    x = np.array([True, False, False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D int32 array
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D int64 array with larger values
    x = np.array([0, 1, 1000000, -1000000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D int32 array
    x = np.zeros((2, 2, 3, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D empty int32 array
    x = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 0D boolean array
    x = np.array(True, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D int64 array with negative and positive values
    x = np.array(
        [[[-5, 4], [3, -2]], [[1, 0], [-1, 6]], [[-10, 20], [30, -40]]],
        dtype=np.int64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_invert"] = bitwise_invert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_invert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_invert'.")


check_valid('jax.numpy.bitwise_invert', generated_inputs['jax.numpy.bitwise_invert'], lib="jax", suffix=0)
