
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive and negative integers
    x = np.array([-10, -1, 0, 1, 10, 42], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D int32 array
    x = np.array([[0, 255], [128, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D boolean array
    x = np.array([True, False, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar int64 array
    x = np.array(-5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D int32 array
    x = np.arange(12, dtype=np.int32).reshape((2, 2, 3)) - 6
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D boolean array
    x = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D int64 array
    x = np.arange(16, dtype=np.int64).reshape((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D int64 array with extreme values
    x = np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D array of int32
    x = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 0D boolean array
    x = np.array(True, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D int64 array
    x = np.random.randint(-1000, 1000, size=(3, 3, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_not'.")


check_valid('jax.numpy.bitwise_not', generated_inputs['jax.numpy.bitwise_not'], lib="jax", suffix=0)
