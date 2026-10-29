
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nansum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs, axis=0, keepdims=False
    a = np.array([1.5, np.nan, 2.5, -3.0, np.nan, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 2: 2D float32 array, axis=1, keepdims=True
    a = np.array(
        [[1.0, np.nan, 3.0], [np.nan, -2.0, 4.0], [0.5, 1.5, np.nan]],
        dtype=np.float32,
    )
    axis = 1
    dtype = np.float32
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 3: 2D float64 array with negatives, axis=0, keepdims=False
    a = np.array(
        [
            [-10.5, 20.0, np.nan],
            [np.nan, -5.5, 12.0],
            [3.0, np.nan, -1.0],
            [7.5, 8.5, 9.5],
        ],
        dtype=np.float64,
    )
    axis = 0
    dtype = np.float64
    keepdims = False
    initial = np.array(-2.5, dtype=np.float64)
    where = np.array(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 4: 3D float32 array, axis=2, keepdims=True
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = 2
    dtype = np.float32
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    where[0, 0, 0] = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 5: 3D float64 array with negative axis, axis=-1, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float64)
    a[a < 0] = np.nan
    axis = -1
    dtype = np.float64
    keepdims = False
    initial = np.array(10.0, dtype=np.float64)
    where = np.ones((3, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 6: 4D float32 array, axis=1, keepdims=True
    a = np.full((2, 3, 2, 2), np.nan, dtype=np.float32)
    a[0, 0, 0, 0] = 3.5
    a[1, 2, 1, 1] = -7.2
    axis = 1
    dtype = np.float32
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 3, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 7: 2D integer array, axis=0, keepdims=False, dtype=int32
    a = np.array([[-1, 2, 3], [4, -5, 6]], dtype=np.int32)
    axis = 0
    dtype = np.int32
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 8: 1D array of all NaNs, axis=0, keepdims=True
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    axis = 0
    dtype = np.float64
    keepdims = True
    initial = np.array(0.0, dtype=np.float64)
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 9: 2D array with broadcastable where condition, axis=-2, keepdims=False
    a = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0]], dtype=np.float32)
    axis = -2
    dtype = np.float32
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([[True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 10: 3D array with float32 input accumulated to float64, axis=0, keepdims=False
    a = np.array(
        [[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]],
        dtype=np.float32,
    )
    axis = 0
    dtype = np.float64
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

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
