
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanstd_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=(0,), keepdims=False, ddof=0, float32
    a = np.array([1.0, np.nan, 3.0, -4.0, 5.0], dtype=np.float32)
    axis = (0,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = np.array([True, True, True, True, False])
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    # Input 2: 2D array, axis=(1,), keepdims=True, ddof=1, float64
    a = np.array(
        [[1.5, -2.5, np.nan], [np.nan, 4.0, 5.5], [-1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    axis = (1,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 3: 2D array, axis=(0, 1), keepdims=False, ddof=0, float32
    a = np.array(
        [
            [-10.0, np.nan, 12.0, 3.0],
            [np.nan, -5.0, 0.0, np.nan],
            [7.0, 8.0, -9.0, 1.0],
        ],
        dtype=np.float32,
    )
    axis = (0, 1)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = ~np.isnan(a)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    # Input 4: 3D array, axis=(0,), keepdims=True, ddof=0, float32
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = (0,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = np.ones(a.shape, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 0,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 5: 3D array, axis=(1, 2), keepdims=False, ddof=1, float64
    a = np.random.randn(3, 4, 5).astype(np.float64)
    a[a < -0.5] = np.nan
    axis = (1, 2)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float64)
    where = np.ones(a.shape, dtype=bool)
    where[0, 0, 0] = False
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 1,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    # Input 6: 4D array, axis=(-1,), keepdims=True, ddof=2, float32
    a = np.random.randn(2, 2, 3, 6).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    axis = (-1,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = np.ones(a.shape, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 2,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 7: 2D array, axis=(0,), keepdims=False, ddof=1, float64, with broadcastable `where`
    a = np.array(
        [[1.0, 2.0, -3.0], [np.nan, 5.0, 6.0], [7.0, -8.0, np.nan]],
        dtype=np.float64,
    )
    axis = (0,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float64)
    where = np.array([[True, True, False]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 1,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    # Input 8: 3D array, axis=(0, 2), keepdims=True, ddof=0, float32
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, :, 1] = np.nan
    axis = (0, 2)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = np.ones(a.shape, dtype=bool)
    where[1, 1, 1] = False
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 0,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 9: 1D array, axis=(-1,), keepdims=True, ddof=1, float64
    a = np.array([np.nan, -2.5, 3.5, -4.5, 5.5, np.nan], dtype=np.float64)
    axis = (-1,)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float64)
    where = np.array([True, True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 10: 4D array, axis=(1, 3), keepdims=False, ddof=0, float32
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 4)).astype(np.float32)
    a[1, 1, 0, 2] = np.nan
    axis = (1, 3)
    mean = np.nanmean(a, axis=axis, keepdims=True).astype(np.float32)
    where = np.ones(a.shape, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanstd_2"] = nanstd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanstd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanstd_2'.")


check_valid('jax.numpy.nanstd', generated_inputs['jax.numpy.nanstd_2'], lib="jax", suffix=2)
