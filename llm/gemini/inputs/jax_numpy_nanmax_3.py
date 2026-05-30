
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, float32, with NaNs and negatives
    a = np.array([[8, np.nan, 4, 6], [np.nan, -2, np.nan, -4], [-2, 1, 7, np.nan]], dtype=np.float32)
    axis = (1,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, False, True, False], [False, True, False, True], [True, True, True, False]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 2: 1D array, float64
    a = np.array([1.0, np.nan, -3.0, 10.0, np.nan], dtype=np.float64)
    axis = (0,)
    keepdims = False
    initial = np.array(-5.0, dtype=np.float64)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 3: 3D array, float32, reduction over multiple axes
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    axis = (0, 2)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.random.choice([True, False], size=a.shape).astype(bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 4: 2D array, broadcasting 'where'
    a = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([[True], [False]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 5: 4D array, float64, empty axis list
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    a[0, 0, 0, 0] = np.nan
    axis = ()
    keepdims = True
    initial = np.array(-100.0, dtype=np.float64)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 6: Negative axis
    a = np.array([[np.nan, 2.0], [-5.0, np.nan]], dtype=np.float32)
    axis = (-1,)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 7: All NaNs in array, check behavior with initial and where
    a = np.full((3, 3), np.nan, dtype=np.float32)
    axis = (1,)
    keepdims = True
    initial = np.array(-999.0, dtype=np.float32)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 8: 1D array, large size, float32
    a = np.arange(100, dtype=np.float32)
    a[50] = np.nan
    axis = (0,)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = (a < 80)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 9: 3D array, axis=list of 1 element
    a = np.random.uniform(-10, 10, (2, 2, 2)).astype(np.float32)
    axis = (2,)
    keepdims = True
    initial = np.array(-5.0, dtype=np.float32)
    where = np.array([[[True, False], [True, True]], [[False, True], [True, False]]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 10: 2D float64 with large values
    a = np.array([[1e10, np.nan], [np.nan, -1e10]], dtype=np.float64)
    axis = (0,)
    keepdims = False
    initial = np.array(0.0, dtype=np.float64)
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.nanmax_3"] = nanmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmax_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmax_3'.")


check_valid('jax.numpy.nanmax', generated_inputs['jax.numpy.nanmax_3'], lib="jax", suffix=3)
