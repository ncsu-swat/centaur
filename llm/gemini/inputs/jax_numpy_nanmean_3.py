
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmean_inputs():
    list_of_inputs = []

    # Helper function to generate arrays with some NaNs
    def make_nan_array(shape, dtype=np.float32):
        arr = np.random.randn(*shape).astype(dtype)
        # Introduce NaNs into the array randomly
        mask = np.random.choice([True, False], size=shape, p=[0.2, 0.8])
        arr[mask] = np.nan
        return arr

    # Input 1: 2D array, float32, axis=(0,), keepdims=True
    a = make_nan_array((4, 5), np.float32)
    where = np.random.choice([True, False], size=(4, 5), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float32),
        "keepdims": True,
        "where": where
    })

    # Input 2: 2D array, float64, axis=(1,), keepdims=False
    a = make_nan_array((3, 6), np.float64)
    where = np.random.choice([True, False], size=(3, 6), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (1,),
        "dtype": np.dtype(np.float64),
        "keepdims": False,
        "where": where
    })

    # Input 3: 3D array, float32, axis=(0, 2), keepdims=True
    a = make_nan_array((2, 3, 4), np.float32)
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0, 2),
        "dtype": np.dtype(np.float32),
        "keepdims": True,
        "where": where
    })

    # Input 4: 1D array, float32, axis=(0,), keepdims=False
    a = make_nan_array((10,), np.float32)
    where = np.random.choice([True, False], size=(10,), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where
    })

    # Input 5: 4D array, float64, axis=(1, 3), keepdims=True
    a = make_nan_array((2, 2, 3, 3), np.float64)
    where = np.random.choice([True, False], size=(2, 2, 3, 3), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (1, 3),
        "dtype": np.dtype(np.float64),
        "keepdims": True,
        "where": where
    })

    # Input 6: 2D array with negative values, float32, axis=(0,), keepdims=False
    a = make_nan_array((5, 5), np.float32) - 5.0
    where = np.random.choice([True, False], size=(5, 5), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where
    })

    # Input 7: 3D array, float32, axis=(1,), keepdims=True
    a = make_nan_array((3, 3, 3), np.float32)
    where = np.random.choice([True, False], size=(3, 3, 3), p=[0.8, 0.2])
    list_of_inputs.append({
        "a": a,
        "axis": (1,),
        "dtype": np.dtype(np.float32),
        "keepdims": True,
        "where": where
    })

    # Input 8: 2D array, float64, axis=(0, 1), keepdims=False
    a = make_nan_array((4, 4), np.float64)
    where = np.random.choice([True, False], size=(4, 4), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0, 1),
        "dtype": np.dtype(np.float64),
        "keepdims": False,
        "where": where
    })

    # Input 9: 1D array, float64, axis=(0,), keepdims=True
    a = make_nan_array((15,), np.float64)
    where = np.random.choice([True, False], size=(15,), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float64),
        "keepdims": True,
        "where": where
    })

    # Input 10: 3D array, float32, axis=(2,), keepdims=False
    a = make_nan_array((2, 4, 3), np.float32)
    where = np.random.choice([True, False], size=(2, 4, 3), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": (2,),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.nanmean_3"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_3'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_3'], lib="jax", suffix=3)
