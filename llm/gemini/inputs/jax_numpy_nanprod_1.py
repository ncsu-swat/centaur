
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanprod_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float32)
    axis = 0
    dtype = np.dtype('float32')
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 2
    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, 8.0]]], dtype=np.float64)
    axis = 1
    dtype = np.dtype('float64')
    keepdims = True
    initial = np.array(2.0, dtype=np.float64)
    where = np.array([[[True, False], [True, True]], [[False, True], [True, True]]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 3
    a = np.array([np.nan, -1.0, 2.0, np.nan, -3.0], dtype=np.float32)
    axis = 0
    dtype = np.dtype('float32')
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 4
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    axis = -1
    dtype = np.dtype('float64')
    keepdims = True
    initial = np.array(1.5, dtype=np.float64)
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 5
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    axis = 2
    dtype = np.dtype('float32')
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 6
    a = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan], [7.0, 8.0], [np.nan, np.nan]], dtype=np.float32)
    axis = -2
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(3.0, dtype=np.float32)
    where = np.array([[True, True], [False, True], [True, False], [True, True], [False, False]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 7
    a = np.array([[[np.nan, 1.0], [2.0, np.nan]], [[3.0, 4.0], [np.nan, np.nan]]], dtype=np.float64)
    axis = 2
    dtype = np.dtype('float64')
    keepdims = False
    initial = np.array(1.0, dtype=np.float64)
    where = np.array([[[True, True], [True, True]], [[True, True], [True, True]]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 8
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, np.nan], dtype=np.float32)
    axis = -1
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([True, True, True, True, True, False, False, True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 9
    a = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [np.nan, 8.0, 9.0]], dtype=np.float64)
    axis = 1
    dtype = np.dtype('float64')
    keepdims = False
    initial = np.array(0.5, dtype=np.float64)
    where = np.array([[True, True, True], [False, True, True], [True, False, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 10
    a = np.array([[[1.0, 2.0, np.nan, 4.0]], [[np.nan, 6.0, 7.0, np.nan]], [[9.0, np.nan, 11.0, 12.0]]], dtype=np.float32)
    axis = 0
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[[True, True, True, True]], [[True, True, True, True]], [[True, True, True, True]]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.nanprod_1"] = nanprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanprod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanprod_1'.")


check_valid('jax.numpy.nanprod', generated_inputs['jax.numpy.nanprod_1'], lib="jax", suffix=1)
