
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmean_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    where = np.array([True, True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 2: 2D array with negative values along axis 1
    a = np.array(
        [[-2.5, np.nan, 1.5], [np.nan, -0.5, -4.0]], dtype=np.float64
    )
    axis = 1
    dtype = np.float64
    keepdims = True
    where = np.array([[True, True, False], [True, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 3: 2D array, reduction along axis 0
    a = np.array([[np.nan, 2.0], [3.0, np.nan], [5.0, 6.0]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    where = np.ones((3, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 4: 3D array with negative axis
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = -1
    dtype = np.float32
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 5: Broadcastable `where` mask with 3D array
    a = np.random.randn(3, 4, 5).astype(np.float64)
    a[a < 0] = np.nan
    axis = 1
    dtype = np.float64
    keepdims = False
    where = np.array([[[True]], [[False]], [[True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 6: 4D array with negative axis -2
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    axis = -2
    dtype = np.float32
    keepdims = False
    where = a > 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 7: All elements NaN along a slice
    a = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    keepdims = True
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 8: Integer input array converted via dtype=np.float32
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    axis = 0
    dtype = np.float32
    keepdims = True
    where = np.array([[True, False, True], [False, True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 9: 1D array with where mask all False
    a = np.array([10.5, 20.2, np.nan, -5.0], dtype=np.float64)
    axis = 0
    dtype = np.float64
    keepdims = False
    where = np.zeros(4, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    # Input 10: 3D array with broadcastable 2D where mask
    a = np.ones((2, 3, 4), dtype=np.float32)
    a[0, :, :] = np.nan
    axis = 2
    dtype = np.float64
    keepdims = False
    where = np.array(
        [[True, False, True, True], [False, True, True, False], [True, True, False, True]],
        dtype=bool,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmean_1"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_1'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_1'], lib="jax", suffix=1)
