
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanvar_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis=0, ddof=0, keepdims=False
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array([True, True, True, False, True, True], dtype=bool)
    mean = np.array([3.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 2: 2D float64 array, axis=0, ddof=1, keepdims=True
    a = np.array(
        [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0], [-1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    axis = 0
    dtype = np.dtype(np.float64)
    ddof = 1
    keepdims = True
    where = np.ones(a.shape, dtype=bool)
    mean = np.array([[2.75, 3.33333333, 5.66666667]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 3: 2D float32 array, axis=1, ddof=0, keepdims=False, with negative values
    a = np.array(
        [
            [-5.0, np.nan, 3.0, 12.0],
            [np.nan, -2.0, -8.0, 4.0],
            [1.0, 2.0, 3.0, np.nan],
        ],
        dtype=np.float32,
    )
    axis = 1
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array(
        [[True, True, True, True], [True, True, False, True], [True, True, True, True]],
        dtype=bool,
    )
    mean = np.array([[3.3333333], [1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 4: 3D float32 array, axis=2, ddof=1, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = 2
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 5: 3D float64 array, axis=-1, ddof=0, keepdims=False
    a = np.random.randn(3, 2, 5).astype(np.float64)
    a[a < 0] = np.nan
    a[np.isnan(a)] = -1.0  # add negative values back
    a[0, 0, 0] = np.nan
    axis = -1
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = False
    where = np.ones((3, 2, 5), dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 6: 3D float32 array, axis=0, ddof=2, keepdims=True
    a = np.random.randn(6, 3, 2).astype(np.float32)
    a[1, 1, 1] = np.nan
    axis = 0
    dtype = np.dtype(np.float32)
    ddof = 2
    keepdims = True
    where = np.ones((6, 3, 2), dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 7: 4D float32 array, axis=1, ddof=0, keepdims=False
    a = np.random.randn(2, 4, 3, 2).astype(np.float32)
    a[0, 2, 1, 0] = np.nan
    axis = 1
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones((2, 4, 3, 2), dtype=bool)
    where[0, 0, 0, 0] = False
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 8: 2D float64 array, axis=-2, ddof=1, keepdims=True
    a = np.array(
        [[np.nan, 2.5, -3.1], [4.2, -1.0, np.nan], [0.0, 7.8, -9.2], [3.3, 1.1, 4.4]],
        dtype=np.float64,
    )
    axis = -2
    dtype = np.dtype(np.float64)
    ddof = 1
    keepdims = True
    where = np.ones((4, 3), dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 9: 1D float64 array, axis=-1, ddof=1, keepdims=True
    a = np.array([-10.5, 0.0, np.nan, 15.2, -2.3, np.nan, 8.1], dtype=np.float64)
    axis = -1
    dtype = np.dtype(np.float64)
    ddof = 1
    keepdims = True
    where = np.array([True, True, True, True, True, False, True], dtype=bool)
    mean = np.array([2.1], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    # Input 10: 4D float32 array, axis=3, ddof=0, keepdims=True
    a = np.random.randn(2, 2, 3, 5).astype(np.float32)
    a[0, 1, 2, 3] = np.nan
    axis = 3
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = True
    where = np.ones((2, 2, 3, 5), dtype=bool)
    mean = np.nanmean(a, axis=axis, keepdims=True)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "ddof": ddof,
                "keepdims": keepdims,
                "where": where,
                "mean": mean,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanvar_1"] = nanvar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanvar_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanvar_1'.")


check_valid('jax.numpy.nanvar', generated_inputs['jax.numpy.nanvar_1'], lib="jax", suffix=1)
