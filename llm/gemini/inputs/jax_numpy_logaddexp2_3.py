
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Scalar numpy integers
    input_dict = {
        "x1": np.int32(5),
        "x2": np.int32(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of int32
    input_dict = {
        "x1": np.array([1, -2, 3], dtype=np.int32),
        "x2": np.array([2, 5, -1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays of int64
    input_dict = {
        "x1": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "x2": np.array([[4, 3], [2, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 1D to 2D
    input_dict = {
        "x1": np.array([1, 2, 3], dtype=np.int32),
        "x2": np.array([[1], [2]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integers (1D array)
    input_dict = {
        "x1": np.array([-10, -20, -30], dtype=np.int32),
        "x2": np.array([-5, -15, -25], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large integers (int64)
    input_dict = {
        "x1": np.array([1000, 2000], dtype=np.int64),
        "x2": np.array([1001, 1999], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mix of zero and positive integers
    input_dict = {
        "x1": np.array([0, 0, 0], dtype=np.int32),
        "x2": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of int32
    input_dict = {
        "x1": np.arange(8, dtype=np.int32).reshape(2, 2, 2),
        "x2": (np.arange(8, dtype=np.int32).reshape(2, 2, 2) * -1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.int16 scalars (negative and positive)
    input_dict = {
        "x1": np.int16(-5),
        "x2": np.int16(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimensional broadcasting
    input_dict = {
        "x1": np.ones((2, 1, 3), dtype=np.int32),
        "x2": np.ones((1, 4, 3), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.logaddexp2_3"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_3'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_3'], lib="jax", suffix=3)
