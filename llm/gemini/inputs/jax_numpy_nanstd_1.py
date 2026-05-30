
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanstd_inputs():
    list_of_inputs = []

    # 1. 2D array, float32, axis=0, ddof=0, keepdims=False
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    mean = np.nanmean(a, axis=0, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # 2. 2D array, float64, axis=1, ddof=1, keepdims=True
    a = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float64)
    where = np.array([[False, True, True], [True, True, False]], dtype=bool)
    mean = np.nanmean(a, axis=1, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 1,
        "dtype": np.float64,
        "ddof": 1,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # 3. 3D array, float32, axis=2, ddof=0, keepdims=False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 3] = np.nan
    where = np.random.choice([True, False], size=a.shape).astype(bool)
    mean = np.nanmean(a, axis=2, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 2,
        "dtype": np.float32,
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # 4. 3D array, float64, axis=0, ddof=1, keepdims=True
    a = np.random.randn(3, 2, 2).astype(np.float64)
    a[1, 0, 1] = np.nan
    where = np.ones(a.shape, dtype=bool)
    mean = np.nanmean(a, axis=0, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 0,
        "dtype": np.float64,
        "ddof": 1,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # 5. 4D array, float32, axis=1, ddof=2, keepdims=False
    a = np.random.randn(2, 4, 3, 2).astype(np.float32)
    a[0, 2, 1, 0] = np.nan
    where = np.random.choice([True, False], size=a.shape).astype(bool)
    mean = np.nanmean(a, axis=1, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 1,
        "dtype": np.float32,
        "ddof": 2,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # 6. 4D array, float64, axis=3, ddof=0, keepdims=True
    a = np.random.randn(2, 2, 2, 5).astype(np.float64)
    a[1, 1, 1, 4] = np.nan
    where = np.ones(a.shape, dtype=bool)
    mean = np.nanmean(a, axis=3, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 3,
        "dtype": np.float64,
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # 7. 2D array, float32, axis=-1, ddof=1, keepdims=False
    a = np.random.randn(5, 5).astype(np.float32)
    a[2, 2] = np.nan
    where = np.random.choice([True, False], size=a.shape).astype(bool)
    mean = np.nanmean(a, axis=-1, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": -1,
        "dtype": np.float32,
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # 8. 3D array, float32, axis=-2, ddof=0, keepdims=True
    a = np.random.randn(3, 4, 5).astype(np.float32)
    a[0, 0, 0] = np.nan
    where = np.ones(a.shape, dtype=bool)
    mean = np.nanmean(a, axis=-2, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": -2,
        "dtype": np.float32,
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean
    })

    # 9. 1D array, float32, axis=0, ddof=0, keepdims=False
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    mean = np.nanmean(a, axis=0, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

    # 10. 5D array, float64, axis=2, ddof=1, keepdims=False
    a = np.random.randn(2, 2, 3, 2, 2).astype(np.float64)
    a[0, 0, 1, 0, 0] = np.nan
    where = np.ones(a.shape, dtype=bool)
    mean = np.nanmean(a, axis=2, keepdims=True)
    list_of_inputs.append({
        "a": a,
        "axis": 2,
        "dtype": np.float64,
        "ddof": 1,
        "keepdims": False,
        "where": where,
        "mean": mean
    })

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
