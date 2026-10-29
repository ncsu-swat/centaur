
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmean_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs
    a = np.array([1.0, np.nan, 3.0, -4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 2: 2D float64 array, reduce along axis 1 with keepdims=True
    a = np.array(
        [[np.nan, 2.5, -1.0], [4.0, np.nan, np.nan], [0.0, 1.2, 3.4]],
        dtype=np.float64,
    )
    axis = (1,)
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 3: 2D float32 array with broadcasted 2D where mask
    a = np.array([[-10.0, np.nan, 5.0], [2.0, -3.0, np.nan]], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array([[True, True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 4: 3D float32 array, multi-axis reduction
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = (0, 2)
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 5: 3D float64 array, reduce along axis 1 with broadcastable mask
    a = np.array(
        [[[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, np.nan], [-1.0, 0.0, 2.0]],
         [[np.nan, 2.0, 3.0], [4.0, np.nan, -6.0], [7.0, -8.0, 9.0], [1.0, 2.0, np.nan]]],
        dtype=np.float64,
    )
    axis = (1,)
    dtype = np.dtype(np.float64)
    keepdims = False
    where = np.array([[[True], [False], [True], [True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 6: 4D float32 array, reduce trailing axes
    a = np.ones((2, 2, 3, 3), dtype=np.float32) * -1.5
    a[0, 0, 1, 1] = np.nan
    axis = (2, 3)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 2, 3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 7: 2D array, reduce all axes
    a = np.array([[np.nan, -2.0], [3.0, 4.0]], dtype=np.float32)
    axis = (0, 1)
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 8: 1D complex64 array
    a = np.array([1.0 + 2.0j, np.nan + 1.0j, -3.0 - 4.0j], dtype=np.complex64)
    axis = (0,)
    dtype = np.dtype(np.complex64)
    keepdims = True
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 9: 3D float32 array, negative axis index
    a = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[np.nan, -5.0], [6.0, np.nan]]],
        dtype=np.float32,
    )
    axis = (-1,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 10: 2D float32 array with boolean mask
    a = np.arange(25, dtype=np.float32).reshape((5, 5)) - 10.0
    a[2, :] = np.nan
    axis = (1,)
    dtype = np.dtype(np.float32)
    keepdims = True
    where = a > -5.0
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    # Input 11: 4D float64 array, reduce leading 3 axes
    a = np.arange(48, dtype=np.float64).reshape((2, 3, 2, 4)) - 20.0
    a[0, 0, 0, 0] = np.nan
    axis = (0, 1, 2)
    dtype = np.dtype(np.float64)
    keepdims = False
    where = np.ones((2, 3, 2, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "where": where}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmean_3"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_3'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_3'], lib="jax", suffix=3)
