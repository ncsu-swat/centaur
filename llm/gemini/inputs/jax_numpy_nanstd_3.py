
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanstd_inputs():
    list_of_inputs = []

    # Input 1: 2D array, reduce axis (0,), float32, keepdims=False
    a = np.array(
        [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [np.nan, 8.0, 9.0]],
        dtype=np.float32,
    )
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    mean = np.nanmean(a, axis=axis, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reduce axis (1,), float64, keepdims=True, ddof=1
    a = np.array(
        [[-2.5, 3.1, np.nan, -0.4], [10.0, -1.2, 5.5, np.nan]], dtype=np.float64
    )
    axis = (1,)
    dtype = np.dtype(np.float64)
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, reduce axis (0,), float32, keepdims=False
    a = np.array([10.0, np.nan, -5.0, 20.0, 0.0, np.nan], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array([True, True, True, False, True, True], dtype=bool)
    mean = np.array([1.25], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, reduce multiple axes (0, 2), float32, keepdims=True
    a = np.random.randn(3, 4, 5).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[2, 3, 4] = np.nan
    axis = (0, 2)
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, reduce axis (1,), float64, keepdims=False
    a = np.linspace(-10, 10, 24, dtype=np.float64).reshape(2, 3, 4)
    a[0, 0, 0] = np.nan
    axis = (1,)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = False
    where = (a > -5.0) | np.isnan(a)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, reduce axes (1, 3), float32, keepdims=True
    a = np.random.uniform(-1.0, 1.0, size=(2, 3, 2, 4)).astype(np.float32)
    a[1, 1, 1, 1] = np.nan
    axis = (1, 3)
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, reduce axis (-1,), float32, keepdims=False
    a = np.array(
        [[np.nan, 2.0], [3.0, 4.0], [np.nan, np.nan]], dtype=np.float32
    )
    axis = (-1,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[2.0], [3.5], [0.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with negative numbers, reduce axis (0, 1), float64
    a = -np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    a[0, 1, 1] = np.nan
    axis = (0, 1)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    where[1, 0, 0] = False
    mean = np.nanmean(a, axis=axis, keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, reduce axis (0,), float64, keepdims=True
    a = np.array([np.nan, np.nan, 42.0, -42.0, 0.0], dtype=np.float64)
    axis = (0,)
    dtype = np.dtype(np.float64)
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.array([0.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, reduce all axes (0, 1, 2, 3), float32, keepdims=False
    a = np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0
    a[0, 0, 0, 0] = np.nan
    axis = (0, 1, 2, 3)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[[[5.0]]]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanstd_3"] = nanstd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanstd_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanstd_3'.")


check_valid('jax.numpy.nanstd', generated_inputs['jax.numpy.nanstd_3'], lib="jax", suffix=3)
