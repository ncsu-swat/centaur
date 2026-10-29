
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, axis 0
    arrays = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, negative axis -1
    arrays = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, axis 0
    arrays = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array, axis 1
    arrays = np.random.randint(-50, 50, size=(3, 2, 5)).astype(np.int32)
    input_dict = {
        "arrays": arrays,
        "axis": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis -1
    arrays = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D boolean array, negative axis -2
    arrays = np.random.choice([True, False], size=(3, 2, 2)).astype(np.bool_)
    input_dict = {
        "arrays": arrays,
        "axis": -2,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, axis 0
    arrays = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int32 array, axis 2
    arrays = np.random.randint(0, 100, size=(3, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "arrays": arrays,
        "axis": 2,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, negative axis -3
    arrays = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": -3,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 array, axis 0
    arrays = np.random.randint(-100, 100, size=(5, 1)).astype(np.int32)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D float32 array, axis 3
    arrays = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": 3,
        "dtype": np.dtype(np.float32),
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
