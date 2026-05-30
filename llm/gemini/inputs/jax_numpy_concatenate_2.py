
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D array, concat along axis 0
    input_dict = {
        "arrays": np.random.randn(2, 3).astype(np.float32),
        "axis": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, concat along axis 1
    input_dict = {
        "arrays": np.random.randn(3, 4, 5).astype(np.float32),
        "axis": 1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, concat along axis 0
    input_dict = {
        "arrays": np.random.randint(0, 10, size=(2, 2)).astype(np.int32),
        "axis": 0,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, negative axis
    input_dict = {
        "arrays": np.random.randn(4, 2).astype(np.float64),
        "axis": -1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, negative axis
    input_dict = {
        "arrays": np.random.randn(3, 3).astype(np.float32),
        "axis": -1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, casting to float16
    input_dict = {
        "arrays": np.random.randn(2, 2, 2).astype(np.float32),
        "axis": 1,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array, larger first dimension
    input_dict = {
        "arrays": np.ones((5, 10), dtype=np.int64),
        "axis": 0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis in the middle
    input_dict = {
        "arrays": np.zeros((3, 4, 5, 6), dtype=np.float32),
        "axis": 2,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, type promotion via dtype arg
    input_dict = {
        "arrays": np.random.randn(2, 3, 4).astype(np.float32),
        "axis": 1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arange array, type promotion to int64
    input_dict = {
        "arrays": np.arange(6).reshape(2, 3).astype(np.int32),
        "axis": 0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.concatenate_2"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.concatenate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.concatenate_2'.")


check_valid('jax.numpy.concatenate', generated_inputs['jax.numpy.concatenate_2'], lib="jax", suffix=2)
