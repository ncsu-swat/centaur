
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: Python int (zero)
    list_of_inputs.append({"x": 0})

    # Input 2: Python int (positive)
    list_of_inputs.append({"x": 1})

    # Input 3: Python int (negative)
    list_of_inputs.append({"x": -1})

    # Input 4: np.int32 scalar
    list_of_inputs.append({"x": np.int32(0)})

    # Input 5: np.int64 scalar
    list_of_inputs.append({"x": np.int64(2)})

    # Input 6: np.int16 1D array
    list_of_inputs.append({"x": np.array([0, 0, 0], dtype=np.int16)})

    # Input 7: np.int32 1D array with mixed values
    list_of_inputs.append({"x": np.array([-1, 0, 1], dtype=np.int32)})

    # Input 8: np.int64 2D array
    list_of_inputs.append({"x": np.array([[0, 1], [-1, 0]], dtype=np.int64)})

    # Input 9: np.uint8 scalar
    list_of_inputs.append({"x": np.uint8(0)})

    # Input 10: np.int8 3D array
    list_of_inputs.append({"x": np.array([[[0], [1]], [[-1], [0]]], dtype=np.int8)})

    return list_of_inputs

generated_inputs["jax.numpy.atanh_3"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_3'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_3'], lib="jax", suffix=3)
