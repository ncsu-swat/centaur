
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def stack_inputs():
    list_of_inputs = []

    # Case 1: 1D array, axis 0, float32
    arrays = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D array, axis 1, float64
    arrays = np.random.randn(2, 3).astype(np.float64)
    input_dict = {
        "arrays": arrays,
        "axis": 1,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D array, axis -1, int32
    arrays = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D array, negative axis, uint8
    arrays = np.random.randint(0, 255, size=(5, 5)).astype(np.uint8)
    input_dict = {
        "arrays": arrays,
        "axis": -2,
        "dtype": np.dtype('uint8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D array, axis 2, float16
    arrays = np.random.randn(2, 2, 2, 2).astype(np.float16)
    input_dict = {
        "arrays": arrays,
        "axis": 2,
        "dtype": np.dtype('float16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D array, negative axis, int64
    arrays = np.array([10, 20]).astype(np.int64)
    input_dict = {
        "arrays": arrays,
        "axis": -1,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D array, axis 0, bool
    arrays = np.random.choice([True, False], size=(3, 2, 2))
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype('bool')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 5D array, axis 4, float32
    arrays = np.random.randn(2, 1, 2, 1, 2).astype(np.float32)
    input_dict = {
        "arrays": arrays,
        "axis": 4,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 2D array, axis 0, complex64
    arrays = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "arrays": arrays,
        "axis": 0,
        "dtype": np.dtype('complex64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 3D array, axis 1, int16
    arrays = np.random.randint(-100, 100, size=(2, 4, 3)).astype(np.int16)
    input_dict = {
        "arrays": arrays,
        "axis": 1,
        "dtype": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.stack_2"] = stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.stack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.stack_2'.")


check_valid('jax.numpy.stack', generated_inputs['jax.numpy.stack_2'], lib="jax", suffix=2)
