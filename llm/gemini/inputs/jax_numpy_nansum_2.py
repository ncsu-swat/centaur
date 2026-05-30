
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, axis is (0,), keepdims=True, initial=0
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [-1.0, -2.0, -3.0]], dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True], [True, True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype("float32"),
        "keepdims": True,
        "initial": 0,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, float64, axis is (0, 2), keepdims=False, initial=10
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[np.nan, -5.0], [-6.0, 7.0]]], dtype=np.float64)
    where = np.array([[[True, False], [True, True]], [[False, True], [True, True]]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "dtype": np.dtype("float64"),
        "keepdims": False,
        "initial": 10,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, float32, axis is (0,), keepdims=False, initial=-5
    a = np.array([10.0, np.nan, -20.0, 30.0, np.nan], dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype("float32"),
        "keepdims": False,
        "initial": -5,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, float32, axis is (1, 3), keepdims=True, initial=1
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    a[a > 1] = np.nan
    where = np.random.choice([True, False], size=(2, 2, 3, 3))
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "dtype": np.dtype("float32"),
        "keepdims": True,
        "initial": 1,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with all NaNs, float64, axis is (1,), keepdims=False, initial=0
    a = np.full((3, 3), np.nan, dtype=np.float64)
    where = np.ones((3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype("float64"),
        "keepdims": False,
        "initial": 0,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, float32, axis is (0, 1), keepdims=False, initial=100
    a = np.array([[1.5, 2.5], [np.nan, -3.5]], dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "dtype": np.dtype("float32"),
        "keepdims": False,
        "initial": 100,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, float64, axis is (1,), keepdims=True, initial=-1
    a = np.random.randn(2, 4, 2).astype(np.float64)
    a[a < -0.5] = np.nan
    where = np.ones((2, 4, 2), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype("float64"),
        "keepdims": True,
        "initial": -1,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, float64, axis is (0,), keepdims=True, initial=5
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    where = np.array([False, False, False], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype("float64"),
        "keepdims": True,
        "initial": 5,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, large scale values, float32, axis is (0,), keepdims=False, initial=-500
    a = np.array([[1000.0, np.nan], [-2000.0, 3000.0]], dtype=np.float32)
    where = np.array([[True, True], [False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype("float32"),
        "keepdims": False,
        "initial": -500,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, float32, axis is (2,), keepdims=False, initial=2
    a = np.array([[[np.nan, 1.0], [2.0, np.nan]], [[3.0, 4.0], [np.nan, np.nan]]], dtype=np.float32)
    where = np.array([[[True, True], [True, False]], [[False, True], [True, True]]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (2,),
        "dtype": np.dtype("float32"),
        "keepdims": False,
        "initial": 2,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nansum_2'.")


check_valid('jax.numpy.nansum', generated_inputs['jax.numpy.nansum_2'], lib="jax", suffix=2)
