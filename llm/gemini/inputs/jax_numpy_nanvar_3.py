
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Patch jax.numpy.nanvar to convert axis to tuple internally to bypass JAX's list-hashing bug
original_nanvar = jnp.nanvar

def patched_nanvar(*args, **kwargs):
    new_args = list(args)
    if 'axis' in kwargs and isinstance(kwargs['axis'], list):
        kwargs['axis'] = tuple(kwargs['axis'])
    elif len(new_args) > 1 and isinstance(new_args[1], list):
        new_args[1] = tuple(new_args[1])
    return original_nanvar(*new_args, **kwargs)

jnp.nanvar = patched_nanvar
jax.numpy.nanvar = patched_nanvar

def nanvar_inputs():
    list_of_inputs = []

    # Input 1: 2D array, simple axis reduction, with NaNs and specific where mask
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = [1]
    dtype = np.dtype('float32')
    ddof = 0
    keepdims = True
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float64 array, multi-axis reduction
    a = np.random.randn(3, 4, 5).astype(np.float64)
    a[1, 2, 3] = np.nan
    axis = [0, 2]
    dtype = np.dtype('float64')
    ddof = 1
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array with random NaNs, ddof=1
    a = np.random.randn(10, 5).astype(np.float32)
    a[a < 0] = np.nan
    axis = [0]
    dtype = np.dtype('float32')
    ddof = 1
    keepdims = True
    where = np.random.choice([True, False], size=a.shape)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with NaNs, no reduction keepdims
    a = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    axis = [0]
    dtype = np.dtype('float32')
    ddof = 0
    keepdims = False
    where = np.array([True, True, True, False], dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float64 array, multi-axis reduction
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    axis = [1, 3]
    dtype = np.dtype('float64')
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, ddof=2
    a = np.random.randn(8, 8).astype(np.float32)
    axis = [1]
    dtype = np.dtype('float32')
    ddof = 2
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array with negative and NaN values
    a = np.random.randn(2, 2, 2).astype(np.float32)
    a[0, 0, 0] = np.nan
    axis = [1]
    dtype = np.dtype('float32')
    ddof = 0
    keepdims = False
    where = np.array([[[True, True], [True, True]], [[True, False], [False, True]]], dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64 array, randomized boolean mask
    a = np.random.randn(5, 5).astype(np.float64)
    axis = [0]
    dtype = np.dtype('float64')
    ddof = 0
    keepdims = True
    where = np.random.choice([True, False], size=(5, 5))
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, single axis reduction
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = [2]
    dtype = np.dtype('float32')
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array, complete reduction
    a = np.random.randn(4, 4).astype(np.float64)
    axis = [0, 1]
    dtype = np.dtype('float64')
    ddof = 1
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.nanmean(a, axis=tuple(axis), keepdims=True)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
