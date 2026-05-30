
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmax_inputs():
    list_of_inputs = []

    # Input 1: 2D array, axis (0,), with NaNs
    a = np.array([[8.0, np.nan, 4.0, 6.0],
                  [np.nan, -2.0, np.nan, -4.0],
                  [-2.0, 1.0, 7.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([[True, False, True, False],
                      [True, True, True, True],
                      [False, True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 2: 3D array, axis (1, 2)
    a = np.random.randn(2, 3, 2).astype(np.float32)
    a[0, 1, 1] = np.nan
    axis = (1, 2)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.ones((2, 3, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 3: 1D array, float64
    a = np.array([1.5, np.nan, -3.2, 4.8, np.nan], dtype=np.float64)
    axis = (0,)
    keepdims = True
    initial = np.array(-5.0, dtype=np.float64)
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 4: 4D array, axis (0, 2)
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (0, 2)
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 5: 2D array with negative values and NaNs
    a = np.array([[-1.0, -2.0, np.nan, -4.0],
                  [-5.0, np.nan, -7.0, -8.0]], dtype=np.float32)
    axis = (1,)
    keepdims = True
    initial = np.array(-20.0, dtype=np.float32)
    where = np.array([[True, True, False, True],
                      [False, True, True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 6: 2D array, full reduction, axis (0, 1)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 7: 3D array, float64, axis (2,)
    a = np.random.randn(2, 2, 3).astype(np.float64)
    axis = (2,)
    keepdims = True
    initial = np.array(-1e5, dtype=np.float64)
    where = np.ones((2, 2, 3), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 8: 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-50.0, dtype=np.float32)
    where = np.array([True, True, True, True, True, False, False, False, False, False], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 9: 5D array, axis (0, 2, 4)
    a = np.random.randn(2, 1, 2, 1, 2).astype(np.float32)
    axis = (0, 2, 4)
    keepdims = True
    initial = np.array(-99.0, dtype=np.float32)
    where = np.ones((2, 1, 2, 1, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 10: 2D array, broadcastable where
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-5.0, dtype=np.float32)
    # where is broadcast compatible: (3, 1) broadcasted to (3, 2)
    where = np.array([[True], [False], [True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

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
