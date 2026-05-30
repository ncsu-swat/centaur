
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanstd_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, reduction along axis 0
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # Input 2: 2D array, float64, reduction along axis 1, keepdims=True
    a = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0]], dtype=np.float64)
    axis = (1,)
    where = np.array([[True, True, True], [False, True, True]], dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float64'),
        "ddof": 1,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # Input 3: 3D array, float32, multi-axis reduction (0, 2)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 0, 0] = np.nan
    axis = (0, 2)
    where = np.ones_like(a, dtype=bool)
    where[1, 1, 1] = False
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # Input 4: 1D array, float32, keepdims=True
    a = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, False, True], dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # Input 5: 4D array, float64, multi-axis reduction (1, 3)
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    a[0, 1, 0, 1] = np.nan
    axis = (1, 3)
    where = np.random.choice([True, False], size=a.shape)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float64'),
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # Input 6: 2D array, float32, axis 0, ddof=1, keepdims=True
    a = np.array([[-1.0, 2.0], [np.nan, -4.0]], dtype=np.float32)
    axis = (0,)
    where = np.array([[True, True], [True, True]], dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 1,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # Input 7: 3D array, float32, axis 1
    a = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (1,)
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # Input 8: 2D array, float32, all axes reduced
    a = np.random.randn(3, 3).astype(np.float32)
    axis = (0, 1)
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # Input 9: 4D array, float32, axis 2, ddof=2
    a = np.random.randn(2, 2, 4, 2).astype(np.float32)
    axis = (2,)
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float32'),
        "ddof": 2,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # Input 10: 3D array, float64, complete reduction to scalar
    a = np.random.randn(3, 3, 3).astype(np.float64)
    axis = (0, 1, 2)
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True, where=where)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": np.dtype('float64'),
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

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
