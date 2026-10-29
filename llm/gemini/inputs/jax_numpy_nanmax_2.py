
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaN values, reducing along axis (0,)
    a = np.array([1.0, np.nan, 3.0, -2.0, np.nan, 5.0], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.array([True, True, True, True, False, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 2: 2D array, keepdims=True, reducing along axis (1,)
    a = np.array(
        [[np.nan, 2.0, 3.0], [-1.0, -5.0, np.nan], [0.0, np.nan, -10.0]],
        dtype=np.float64,
    )
    axis = (1,)
    keepdims = True
    initial = np.array(-np.inf, dtype=np.float64)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 3: 2D array reducing along multiple axes (0, 1)
    a = np.array([[np.nan, np.nan], [1.5, -3.2]], dtype=np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 4: 3D array, axis (0, 2)
    a = np.array(
        [
            [[1.0, -2.0], [np.nan, 4.0]],
            [[-5.0, np.nan], [3.0, 2.0]],
            [[np.nan, np.nan], [0.0, -1.0]],
        ],
        dtype=np.float32,
    )
    axis = (0, 2)
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([[[True, True], [False, True]]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 5: 1D array with negative axis (-1,)
    a = np.array([-10.0, -20.0, np.nan, -5.0, -1.0], dtype=np.float32)
    axis = (-1,)
    keepdims = False
    initial = np.array(-50.0, dtype=np.float32)
    where = np.array([True, False, True, True, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 6: 4D array, axis (1, 3)
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    a[0, 1, 0, 2] = np.nan
    a[1, 2, 1, 3] = np.nan
    axis = (1, 3)
    keepdims = False
    initial = np.array(-1000.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(2, 3, 2, 4)).astype(bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 7: 2D array with all NaNs
    a = np.full((3, 4), np.nan, dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 8: 2D array, broadcasting where with shape (3, 1)
    a = np.array([[10.0, 20.0, np.nan], [np.nan, 5.0, 1.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    axis = (0,)
    keepdims = False
    initial = np.array(-np.inf, dtype=np.float64)
    where = np.array([[True], [False], [True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 9: 3D array reducing along axis (1,)
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[a % 3 == 0] = np.nan
    axis = (1,)
    keepdims = True
    initial = np.array(-5.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 10: 2D array with negative numbers, axis (-2,)
    a = np.array([[-100.5, -200.5], [np.nan, -50.0], [-10.0, np.nan]], dtype=np.float32)
    axis = (-2,)
    keepdims = False
    initial = np.array(-300.0, dtype=np.float32)
    where = np.array([[True, True], [True, False], [False, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmax_2"] = nanmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmax_2'.")


check_valid('jax.numpy.nanmax', generated_inputs['jax.numpy.nanmax_2'], lib="jax", suffix=2)
