
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def repeat_inputs():
    list_of_inputs = []

    # Input 1: 1D array, perfect match
    a = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": 2,
        "axis": 0,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, repeat along axis 0
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "a": a,
        "repeats": 3,
        "axis": 0,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, repeat along axis 1
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "a": a,
        "repeats": 2,
        "axis": 1,
        "total_repeat_length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, negative axis
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "repeats": 2,
        "axis": -1,
        "total_repeat_length": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger total_repeat_length (padding final value)
    a = np.array([5, 6], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": 2,
        "axis": 0,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Smaller total_repeat_length (truncation)
    a = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": 3,
        "axis": 0,
        "total_repeat_length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 2D array with negative axis
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "repeats": 2,
        "axis": -2,
        "total_repeat_length": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D integer array
    a = np.random.randint(0, 10, size=(2, 2, 2, 2)).astype(np.int64)
    input_dict = {
        "a": a,
        "repeats": 3,
        "axis": 2,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High repeats value
    a = np.array([[1], [2]], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": 10,
        "axis": 1,
        "total_repeat_length": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 1D array
    a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    input_dict = {
        "a": a,
        "repeats": 1,
        "axis": 0,
        "total_repeat_length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.repeat_2"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_2'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_2'], lib="jax", suffix=2)
