
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanvar_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=(0,)
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0], dtype=np.float32)
    where = np.array([True, True, True, False, True, True], dtype=bool)
    axis = (0,)
    mean = np.array([3.5], dtype=np.float32)
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

    # Input 2: 2D array, float64, axis=(1,), ddof=1, keepdims=True
    a = np.array(
        [
            [-2.5, np.nan, 3.5, 4.0],
            [1.0, 2.0, np.nan, -5.0],
            [np.nan, 0.0, 2.5, 6.0],
        ],
        dtype=np.float64,
    )
    where = np.ones_like(a, dtype=bool)
    axis = (1,)
    mean = np.nanmean(a, axis=axis, keepdims=True)
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

    # Input 3: 2D array with negative numbers, axis=(0,), ddof=0, keepdims=False
    a = np.array(
        [
            [-10.0, 20.0, -30.0],
            [np.nan, -5.0, 15.0],
            [40.0, np.nan, -50.0],
            [-60.0, 70.0, np.nan],
        ],
        dtype=np.float32,
    )
    where = np.array(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
            [True, False, True],
        ],
        dtype=bool,
    )
    axis = (0,)
    mean = np.zeros((1, 3), dtype=np.float32)
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

    # Input 4: 3D array, axis=(0, 2), float32, keepdims=True, ddof=1
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.0, scale=2.0, size=(3, 4, 5)).astype(np.float32)
    a[rng.uniform(size=a.shape) < 0.2] = np.nan
    where = rng.choice([True, False], size=a.shape, p=[0.8, 0.2])
    axis = (0, 2)
    mean = np.zeros((1, 4, 1), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 5: 3D array, axis=(1,), float64, keepdims=False, ddof=0
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4)) - 12.0
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = np.ones((2, 3, 4), dtype=bool)
    axis = (1,)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    # Input 6: 4D array, axis=(2, 3), float32, keepdims=True, ddof=0
    a = rng.uniform(low=-10.0, high=10.0, size=(2, 2, 3, 4)).astype(np.float32)
    a[0, 0, 1, 1] = np.nan
    a[1, 1, 2, 3] = np.nan
    where = np.ones_like(a, dtype=bool)
    axis = (2, 3)
    mean = np.zeros((2, 2, 1, 1), dtype=np.float32)
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

    # Input 7: 2D array, axis=(-1,), float64, ddof=1, keepdims=False
    a = np.array(
        [[np.nan, 2.0, 3.0, 4.0], [5.0, -6.0, np.nan, 8.0]], dtype=np.float64
    )
    where = np.array(
        [[True, True, True, False], [True, False, True, True]], dtype=bool
    )
    axis = (-1,)
    mean = np.array([[2.5], [5.0]], dtype=np.float64)
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

    # Input 8: 3D array, axis=(-2,), float32, keepdims=True, ddof=0
    a = rng.standard_normal(size=(2, 5, 2)).astype(np.float32)
    a[0, 2, 1] = np.nan
    where = np.ones_like(a, dtype=bool)
    axis = (-2,)
    mean = np.nanmean(a, axis=axis, keepdims=True)
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

    # Input 9: 1D array with negative values, axis=(-1,), float64, keepdims=True, ddof=0
    a = np.array([-100.5, np.nan, -50.2, 0.0, 75.3], dtype=np.float64)
    where = np.array([True, True, True, True, False], dtype=bool)
    axis = (-1,)
    mean = np.array([-50.233333], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float64,
            "ddof": 0,
            "keepdims": True,
            "where": where,
            "mean": mean,
        }
    )

    # Input 10: 4D array, axis=(1,), float32, ddof=2, keepdims=False
    a = rng.normal(loc=5.0, scale=1.0, size=(2, 6, 2, 2)).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    a[1, 5, 1, 1] = np.nan
    where = np.ones((2, 6, 2, 2), dtype=bool)
    axis = (1,)
    mean = np.full((2, 1, 2, 2), 5.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": np.float32,
            "ddof": 2,
            "keepdims": False,
            "where": where,
            "mean": mean,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanvar_3"] = nanvar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanvar_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanvar_3'.")


check_valid('jax.numpy.nanvar', generated_inputs['jax.numpy.nanvar_3'], lib="jax", suffix=3)
