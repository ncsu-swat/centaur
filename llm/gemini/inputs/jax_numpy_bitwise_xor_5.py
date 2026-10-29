
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    input_dict = {
        "x": True,
        "y": np.array([True, False, True, False], dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with positive values
    input_dict = {"x": False, "y": np.array([0, 1, 2, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array with negative values
    input_dict = {
        "x": True,
        "y": np.array([[-1, -2], [3, 4]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar tensor (int64)
    input_dict = {"x": True, "y": np.array(7, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array
    input_dict = {
        "x": False,
        "y": np.arange(8, dtype=np.int32).reshape((2, 2, 2)),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array
    input_dict = {
        "x": False,
        "y": np.array(
            [[True, False, True], [False, True, False]], dtype=np.bool_
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array with mixed values
    input_dict = {
        "x": True,
        "y": np.array([-128, -1, 0, 1, 127], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int64 array
    input_dict = {
        "x": True,
        "y": np.ones((2, 1, 3, 2), dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 array with large values
    input_dict = {
        "x": False,
        "y": np.array([0, 1024, 65535, 2147483647], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int64 array with large negative and positive values
    input_dict = {
        "x": True,
        "y": np.array([[-1000000000, 1000000000], [0, -1]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_xor_5"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_5'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_5'], lib="jax", suffix=5)
