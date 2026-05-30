
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmean_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, keepdims=True, axis=(0,)
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.float32,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64, keepdims=False, axis=(1,)
    a = np.array([[np.nan, 2.0], [3.0, 4.0], [np.nan, np.nan]], dtype=np.float64)
    where = np.array([[True, True], [False, True], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.float64,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32, keepdims=True, axis=(0, 2)
    a = np.random.randn(2, 3, 2).astype(np.float32)
    a[0, 1, 1] = np.nan
    where = np.random.choice([True, False], size=(2, 3, 2)).astype(bool)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "dtype": np.float32,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64, keepdims=False, axis=(0,)
    a = np.array([1.0, np.nan, -3.0, 4.0], dtype=np.float64)
    where = np.array([True, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.float64,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32, keepdims=True, axis=(1, 3)
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[1, 0, 1, 0] = np.nan
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "dtype": np.float32,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32, keepdims=False, axis=(0, 1), with broadcast where
    a = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    where = np.array([[True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64, keepdims=True, axis=(1,)
    a = np.random.randn(3, 1, 4).astype(np.float64)
    where = np.random.choice([True, False], size=(3, 1, 4)).astype(bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.float64,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64, keepdims=False, axis=(0,)
    a = np.random.randn(4, 4).astype(np.float64)
    a[2, 2] = np.nan
    where = np.ones((4, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.float64,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32, keepdims=True, axis=(0,)
    a = np.array([np.nan, np.nan, 3.0, np.nan], dtype=np.float32)
    where = np.array([True, True, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.float32,
        "keepdims": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, keepdims=False, axis=(2,)
    a = np.random.randn(2, 2, 3).astype(np.float32)
    where = np.array([[[True, False, True]], [[True, True, False]]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (2,),
        "dtype": np.float32,
        "keepdims": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanmean_2"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_2'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_2'], lib="jax", suffix=2)
