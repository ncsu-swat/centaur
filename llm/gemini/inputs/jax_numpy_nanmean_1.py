
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmean_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D array with some NaNs, float32
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, float64, negative values, keepdims=True
    a = np.array([[[np.nan, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [np.nan, np.nan]]], dtype=np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float64,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, all NaNs
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    where = np.array([True, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, large float64 values, negative axis
    a = np.array([[1e10, np.nan], [np.nan, -1e10]], dtype=np.float64)
    where = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "dtype": np.float64,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with no NaNs, checking standard mean behaviour
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    where = a % 2 == 0  # Filter to only keep even numbers
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, float32, reducing along axis 3
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": 3,
        "dtype": np.float32,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, where clause masks all elements
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    where = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with float64, single element dimensions
    a = np.array([[np.nan]], dtype=np.float64)
    where = np.array([[True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float64,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, axis=-2, float32, mixed NaNs
    a = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, np.nan], [4.0, 5.0]]], dtype=np.float32)
    where = np.array([[[True, False], [True, True]], [[False, False], [True, True]]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -2,
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, dtype cast to float64, keepdims=False
    a = np.array([[10.5, np.nan, -5.5], [np.nan, 20.0, 30.0]], dtype=np.float32)
    where = np.array([[True, True, True], [True, True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float64,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanmean_1"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_1'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_1'], lib="jax", suffix=1)
