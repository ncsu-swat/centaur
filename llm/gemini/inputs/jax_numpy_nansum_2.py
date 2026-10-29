
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nansum_inputs():
    list_of_inputs = []

    # Input 1: 2D array, sum along axis 0
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    dtype = np.float32
    keepdims = False
    initial = 0
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 2: 2D array, sum along axis 1 with keepdims=True
    a = np.array([[-1.5, np.nan], [np.nan, -2.5], [3.0, 4.0]], dtype=np.float64)
    axis = (1,)
    dtype = np.float64
    keepdims = True
    initial = 10
    where = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 3: 3D array, sum over multiple axes
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 3] = np.nan
    axis = (0, 2)
    dtype = np.float32
    keepdims = False
    initial = -5
    where = a > 5
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 4: 1D array with all NaNs
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    axis = (0,)
    dtype = np.float32
    keepdims = True
    initial = 0
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 5: 3D array, negative axis index
    a = np.random.randn(3, 4, 5).astype(np.float32)
    a[a < 0] = np.nan
    axis = (-1,)
    dtype = np.float32
    keepdims = False
    initial = 2
    where = np.ones((3, 4, 5), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 6: 4D array, float64 dtype
    a = np.zeros((2, 2, 2, 2), dtype=np.float64)
    a[0, 0, 0, 0] = np.nan
    a[1, 1, 1, 1] = 5.0
    axis = (1, 3)
    dtype = np.float64
    keepdims = True
    initial = 1
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 7: 2D array, broadcasting 'where'
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, np.nan]],
        dtype=np.float32,
    )
    axis = (0,)
    dtype = np.float32
    keepdims = False
    initial = 0
    where = np.array([[True, False, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 8: 2D array, negative values, sum across all axes specified in tuple
    a = np.array([[-10.0, np.nan], [-20.0, -30.0]], dtype=np.float32)
    axis = (0, 1)
    dtype = np.float32
    keepdims = False
    initial = -100
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 9: 1D array with positive initial and keepdims=False
    a = np.array([1.5, np.nan, 2.5, np.nan, 3.5], dtype=np.float64)
    axis = (0,)
    dtype = np.float64
    keepdims = False
    initial = 5
    where = np.array([True, True, True, False, False], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 10: 3D array with all elements filtered by where=False
    a = np.array([[[1.0, 2.0], [np.nan, 4.0]]], dtype=np.float32)
    axis = (1, 2)
    dtype = np.float32
    keepdims = True
    initial = 0
    where = np.zeros_like(a, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nansum_2'.")


check_valid('jax.numpy.nansum', generated_inputs['jax.numpy.nansum_2'], lib="jax", suffix=2)
