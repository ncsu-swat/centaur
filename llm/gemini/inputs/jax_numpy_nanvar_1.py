
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanvar_inputs():
    list_of_inputs = []

    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    mean = np.array([[2.5, 5.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.array([[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[[1.0, 3.0]], [[6.0, 8.0]]], dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.random.randn(4, 5).astype(np.float64)
    a[1, 2] = np.nan
    a[3, 4] = np.nan
    axis = 1
    dtype = np.float64
    ddof = 0
    keepdims = False
    where = np.ones((4, 5), dtype=bool)
    mean = np.zeros((4, 1), dtype=np.float64)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    ddof = 1
    keepdims = False
    where = np.array([True, True, True, False, True], dtype=bool)
    mean = np.array([3.33], dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.random.randn(3, 3, 3).astype(np.float32)
    a[a > 1] = np.nan
    axis = 2
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.ones((3, 3, 3), dtype=bool)
    mean = np.zeros((3, 3, 1), dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.random.randn(6, 2).astype(np.float64)
    a[0, 1] = np.nan
    axis = 0
    dtype = np.float64
    ddof = 1
    keepdims = False
    where = np.ones((6, 2), dtype=bool)
    mean = np.zeros((1, 2), dtype=np.float64)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.random.randn(2, 4, 3).astype(np.float32)
    a[0, 0, 0] = np.nan
    axis = 1
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.ones((2, 4, 3), dtype=bool)
    mean = np.zeros((2, 1, 3), dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    mean = np.array([[2.0], [3.5]], dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.random.randn(5, 5).astype(np.float64)
    axis = 0
    dtype = np.float64
    ddof = 1
    keepdims = True
    where = (a > -1.0)
    mean = np.zeros((1, 5), dtype=np.float64)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    a = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    ddof = 1
    keepdims = True
    where = np.array([True, False, True], dtype=bool)
    mean = np.array([1.5], dtype=np.float32)
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype, "ddof": ddof, "keepdims": keepdims, "where": where, "mean": mean})

    return list_of_inputs

generated_inputs["jax.numpy.nanvar_1"] = nanvar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanvar_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanvar_1'.")


check_valid('jax.numpy.nanvar', generated_inputs['jax.numpy.nanvar_1'], lib="jax", suffix=1)
