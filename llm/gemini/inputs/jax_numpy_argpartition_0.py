
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argpartition_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive integers, axis=-1, kth in the middle
    a = np.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "kth": 4,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, random floats including negatives, axis=0, kth=2
    a = np.array([-1.5, 3.2, -0.1, 0.0, 5.5, -10.2], dtype=np.float32)
    input_dict = {
        "a": a,
        "kth": 2,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, partition along axis=0
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "kth": 1,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, int32, partition along axis=1
    a = np.random.randint(-50, 50, size=(4, 6)).astype(np.int32)
    input_dict = {
        "a": a,
        "kth": 3,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, partition along axis=-1
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "kth": 2,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of larger size, float64, axis=0
    a = np.linspace(10.0, -10.0, 20).astype(np.float64)
    input_dict = {
        "a": a,
        "kth": 10,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, int16, partition along axis=2
    a = np.random.randint(-100, 100, size=(2, 3, 4, 2)).astype(np.int16)
    input_dict = {
        "a": a,
        "kth": 1,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, float16, partition along axis=-2
    a = np.random.randn(6, 3).astype(np.float16)
    input_dict = {
        "a": a,
        "kth": 4,
        "axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, kth is positive integer
    a = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "kth": 6,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, axis=1, kth=0 (minimum element partition)
    a = np.random.randn(2, 5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "kth": 0,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argpartition"] = argpartition_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argpartition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argpartition'.")


check_valid('jax.numpy.argpartition', generated_inputs['jax.numpy.argpartition'], lib="jax", suffix=0)
