
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanstd_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, axis=0, basic float32
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float32)
    where = np.array([True, True, True, True, True], dtype=bool)
    mean = np.nanmean(a, axis=0, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, float32, ddof=1, keepdims=True
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype(np.float32),
        "ddof": 1,
        "keepdims": True,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, negative values included, keepdims=False
    a = np.array(
        [[-10.0, np.nan, -3.0], [np.nan, -5.0, -1.0], [-2.0, -8.0, np.nan]],
        dtype=np.float32,
    )
    where = np.array(
        [[True, False, True], [False, True, True], [True, True, False]],
        dtype=bool,
    )
    mean = np.nanmean(np.where(where, a, np.nan), axis=0, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=-1, float32, ddof=2
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = np.ones((2, 3, 4), dtype=bool)
    mean = np.nanmean(a, axis=-1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": -1,
        "dtype": np.dtype(np.float32),
        "ddof": 2,
        "keepdims": True,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=1, keepdims=False, complex64
    a = (
        np.array(
            [[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]],
            dtype=np.float32,
        )
        + 1j
    ).astype(np.complex64)
    where = np.ones((2, 2, 2), dtype=bool)
    mean = np.nanmean(a, axis=1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype(np.complex64),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=2, float32, ddof=1
    a = np.ones((2, 3, 4, 2), dtype=np.float32) * 2.5
    a[0, 0, 1, 0] = np.nan
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=2, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.dtype(np.float32),
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, negative axis index (axis=-2)
    a = np.array(
        [[1.5, -2.5, 3.5], [np.nan, 0.0, -1.5], [4.0, np.nan, 2.0]],
        dtype=np.float32,
    )
    where = np.array(
        [[True, True, True], [True, True, True], [True, True, True]], dtype=bool
    )
    mean = np.nanmean(a, axis=-2, keepdims=True)
    input_dict = {
        "a": a,
        "axis": -2,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with mixed positives and negatives, ddof=1
    a = np.array([-100.0, 50.0, np.nan, 25.0, -25.0], dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    mean = np.nanmean(np.where(where, a, np.nan), axis=0, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype(np.float32),
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with varying elements
    a = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    a[0, 0, 0] = np.nan
    a[1, 1, 2] = np.nan
    where = np.ones((2, 2, 3), dtype=bool)
    mean = np.nanmean(a, axis=0, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, where mask with some False values
    a = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    where = np.array([[True, False, True], [False, True, False]], dtype=bool)
    mean = np.nanmean(np.where(where, a, np.nan), axis=1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanstd_1"] = nanstd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanstd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanstd_1'.")


check_valid('jax.numpy.nanstd', generated_inputs['jax.numpy.nanstd_1'], lib="jax", suffix=1)
