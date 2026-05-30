
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def repeat_inputs():
    list_of_inputs = []

    # Case 1: 1D float array, normal repeat
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    repeats = np.array([2, 1, 3, 0, 2], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 0,
        "total_repeat_length": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D int array, repeat along axis 0
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    repeats = np.array([2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 0,
        "total_repeat_length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D int array, repeat along axis 1
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    repeats = np.array([1, 2, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 1,
        "total_repeat_length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D float array with negative values, axis 0
    a = np.array([[-1.5, -2.5], [3.5, 4.5], [-5.5, -6.5]], dtype=np.float32)
    repeats = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 0,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D float array, axis 1
    a = np.random.randn(2, 2, 2).astype(np.float32)
    repeats = np.array([3, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 1,
        "total_repeat_length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Negative axis
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    repeats = np.array([1, 2, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": -1,
        "total_repeat_length": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Truncated output (total_repeat_length < sum(repeats))
    a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    repeats = np.array([3, 3, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 0,
        "total_repeat_length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Padded output (total_repeat_length > sum(repeats))
    a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    repeats = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 0,
        "total_repeat_length": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 4D array, axis of size 1
    a = np.random.randn(2, 1, 2, 2).astype(np.float32)
    repeats = np.array([3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 1,
        "total_repeat_length": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 3D float64 array, axis 2
    a = np.random.randn(3, 1, 3).astype(np.float64)
    repeats = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "repeats": repeats,
        "axis": 2,
        "total_repeat_length": 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.repeat_1"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_1'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_1'], lib="jax", suffix=1)
