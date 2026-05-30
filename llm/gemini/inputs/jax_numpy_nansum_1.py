
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, basic sum along axis 0
    a = np.array([[1.5, np.nan, 3.0], [np.nan, 4.5, 6.0]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, True, True], [True, True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 2: 1D array with negative values, float64, keeping dims
    a = np.array([-1.0, np.nan, -3.5, 4.0, np.nan], dtype=np.float64)
    axis = 0
    dtype = np.float64
    keepdims = True
    initial = np.array(10.0, dtype=np.float64)
    where = np.array([True, False, True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 3: 3D array, axis 2, casting output to float32
    a = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, 4.0], [5.0, np.nan]]], dtype=np.float64)
    axis = 2
    dtype = np.float32
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([[[True, True], [True, True]], [[True, True], [True, True]]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 4: 2D array, negative axis, mixed NaNs and infs
    a = np.array([[np.inf, np.nan, -np.inf], [2.0, 3.0, np.nan]], dtype=np.float32)
    axis = -1
    dtype = np.float32
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.array([[True, True, False], [False, True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 5: 4D array, axis 1, standard setup
    a = np.random.randn(2, 3, 2, 2).astype(np.float32)
    a[0, 1, 0, 1] = np.nan
    a[1, 2, 1, 0] = np.nan
    axis = 1
    dtype = np.float32
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 6: 2D array, all elements are NaN, expect initial or 0 as sum
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    keepdims = False
    initial = np.array(100.0, dtype=np.float32)
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 7: 1D array, large size, float64
    a = np.arange(100, dtype=np.float64)
    a[::10] = np.nan
    axis = 0
    dtype = np.float64
    keepdims = True
    initial = np.array(-50.0, dtype=np.float64)
    where = (a > 10) & (a < 90)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 8: 3D array, axis 0, with custom float32 outputs
    a = np.array([[[np.nan, 1.0]], [[2.0, np.nan]], [[3.0, 4.0]]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    initial = np.array(0.5, dtype=np.float32)
    where = np.array([[[True, True]], [[True, False]], [[False, True]]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 9: 2D array, negative axis, float64 output
    a = np.array([[10.0, 20.0, np.nan], [np.nan, 30.0, 40.0]], dtype=np.float32)
    axis = -2
    dtype = np.float64
    keepdims = True
    initial = np.array(-10.0, dtype=np.float64)
    where = np.array([[True, False, True], [True, True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    # Input 10: 5D array, axis 3, complex shape, high dimensions
    a = np.zeros((2, 2, 2, 3, 2), dtype=np.float32)
    a[0, 0, 0, :, 0] = [np.nan, 5.0, np.nan]
    axis = 3
    dtype = np.float32
    keepdims = False
    initial = np.array(1.1, dtype=np.float32)
    where = np.ones((2, 2, 2, 3, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, 
        "initial": initial, "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.nansum_1"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nansum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nansum_1'.")


check_valid('jax.numpy.nansum', generated_inputs['jax.numpy.nansum_1'], lib="jax", suffix=1)
