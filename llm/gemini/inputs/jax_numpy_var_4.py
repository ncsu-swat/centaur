
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def var_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, -2.0, 3.5, 4.0, -1.5, 6.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float32)
    correction = 0.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 2: 2D float64 array, reduce axis 1
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = (1,)
    dtype = np.dtype(np.float64)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float64)
    correction = 1.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 3: 2D float32 array, reduce axis 0
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float32)
    correction = 0.5
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 4: 3D float32 array, reduce axes (1, 2)
    a = np.random.randn(2, 4, 5).astype(np.float32)
    axis = (1, 2)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float32)
    correction = 1.5
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 5: 3D float64 array with negative values, reduce axis 0
    a = np.random.uniform(-10.0, 10.0, size=(5, 3, 2)).astype(np.float64)
    axis = (0,)
    dtype = np.dtype(np.float64)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float64)
    correction = 2.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 6: 4D float32 array, reduce axes (0, 2)
    a = np.random.randn(3, 4, 3, 2).astype(np.float32)
    axis = (0, 2)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float32)
    correction = 0.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 7: 2D float32 array with partial boolean mask in `where`
    a = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    axis = (1,)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True, where=where).astype(np.float32)
    correction = 0.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 8: 1D float64 array, negative axis index (-1,)
    a = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
    axis = (-1,)
    dtype = np.dtype(np.float64)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float64)
    correction = 1.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 9: 4D float64 array, reduce axes (1, 3)
    a = np.random.randn(2, 3, 2, 4).astype(np.float64)
    axis = (1, 3)
    dtype = np.dtype(np.float64)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float64)
    correction = 0.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    # Input 10: 3D float32 array, reduce all axes (0, 1, 2)
    a = np.random.uniform(0.0, 100.0, size=(3, 3, 3)).astype(np.float32)
    axis = (0, 1, 2)
    dtype = np.dtype(np.float32)
    ddof = None
    keepdims = False
    where = np.ones(a.shape, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float32)
    correction = 1.0
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "ddof": ddof,
        "keepdims": keepdims, "where": where, "mean": mean, "correction": correction
    })

    return list_of_inputs

generated_inputs["jax.numpy.var_4"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_4'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_4'], lib="jax", suffix=4)
