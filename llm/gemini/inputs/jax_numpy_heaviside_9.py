
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: Scalar integers
    input_dict = {
        "x1": int(0),
        "x2": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of integers, scalar integer
    input_dict = {
        "x1": np.array([-2, -1, 0, 1, 2], dtype=np.int32),
        "x2": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of integers, 1D array of integers
    input_dict = {
        "x1": np.array([-1, 0, 1], dtype=np.int64),
        "x2": np.array([10, 20, 30], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of integers, scalar integer
    input_dict = {
        "x1": np.array([[-3, 0], [0, 3]], dtype=np.int32),
        "x2": int(-1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of integers, 2D array of integers
    input_dict = {
        "x1": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "x2": np.array([[1, 2], [3, 4]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of integers, scalar integer
    input_dict = {
        "x1": np.array([[[-5], [0]], [[5], [10]]], dtype=np.int32),
        "x2": int(100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of integers, broadcast-compatible 1D array of integers
    input_dict = {
        "x1": np.array([-2, 0, 2], dtype=np.int32),
        "x2": np.array([9], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar integer, 2D array of integers
    input_dict = {
        "x1": int(0),
        "x2": np.array([[1, 2], [3, 4]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large random int32 arrays
    input_dict = {
        "x1": np.random.randint(-10, 10, size=(10, 10), dtype=np.int32),
        "x2": np.random.randint(0, 10, size=(10, 10), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 array, scalar int16
    input_dict = {
        "x1": np.array([-10, 0, 10], dtype=np.int16),
        "x2": np.array([42], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_9"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_9'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_9'], lib="jax", suffix=9)
