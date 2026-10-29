
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers, default-like settings
    a = np.array([5, 2, -1, 3, 0, 7], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, sorting along axis 0
    a = np.array(
        [[1.5, -2.3, 0.0], [4.1, -0.5, 3.2], [-1.2, 2.8, -4.0]],
        dtype=np.float32,
    )
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, descending along axis 1, int64 output dtype
    a = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": True,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array with duplicate elements to test stability
    a = np.array(
        [[[3, 1, 3], [2, 2, 1]], [[0, -1, 0], [4, 5, 4]]], dtype=np.int64
    )
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": True,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of negative floats, descending order
    a = np.array([-10.5, -3.2, -0.1, -25.8, -7.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array sorting along middle axis (axis=1)
    a = np.random.uniform(-10.0, 10.0, size=(2, 4, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int16 array sorting along axis 2
    a = np.random.randint(-100, 100, size=(2, 3, 4, 2), dtype=np.int16)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": False,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D uint8 array, descending order along negative axis
    a = np.array([[255, 0, 128], [64, 32, 200]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": True,
        "descending": True,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D boolean array
    a = np.array([True, False, True, False, False, True], dtype=np.bool_)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array sorting along axis 0, unstable sort
    a = np.random.randn(3, 2, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D integer array with single element per axis
    a = np.array([[42]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argsort_1"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_1'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_1'], lib="jax", suffix=1)
