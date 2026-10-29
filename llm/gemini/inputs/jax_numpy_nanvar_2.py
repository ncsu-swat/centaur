
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanvar_inputs():
    list_of_inputs = []

    # Input 1: 2D array, reduce along axis 0, float32
    a = np.array([[1.0, np.nan, 3.0], [-2.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[-0.5, 5.0, 3.0]], dtype=np.float32)
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

    # Input 2: 2D array, reduce along axis 1, keepdims=True, ddof=1, float32
    a = np.array(
        [[10.5, -4.2, np.nan, 8.0], [np.nan, 3.1, -1.0, 0.0]], dtype=np.float32
    )
    axis = (1,)
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = np.array(
        [[True, True, False, True], [False, True, True, True]], dtype=bool
    )
    mean = np.array([[4.7666667], [0.7]], dtype=np.float32)
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

    # Input 3: 1D array, reduce along axis 0
    a = np.array([np.nan, -1.0, 2.5, np.nan, 4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array([True, True, True, True, False], dtype=bool)
    mean = np.array([0.75], dtype=np.float32)
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

    # Input 4: 3D array, reduce along multiple axes (0, 2)
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 0, 0] = np.nan
    axis = (0, 2)
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = a > 5.0
    mean = np.full((1, 3, 1), 14.0, dtype=np.float32)
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

    # Input 5: 3D array, reduce along axis 1, keepdims=False
    a = np.array(
        [[[1.0, 2.0], [np.nan, 4.0]], [[-5.0, -6.0], [7.0, np.nan]]],
        dtype=np.float32,
    )
    axis = (1,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[[1.0, 3.0]], [[1.0, -6.0]]], dtype=np.float32)
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

    # Input 6: 2D array, reduce along all axes (0, 1)
    a = np.array(
        [[-10.0, np.nan, -30.0], [40.0, 50.0, np.nan]], dtype=np.float32
    )
    axis = (0, 1)
    dtype = np.dtype(np.float32)
    ddof = 2
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array([[12.5]], dtype=np.float32)
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

    # Input 7: 4D array, reduce along axis 3
    a = np.zeros((2, 2, 2, 3), dtype=np.float32)
    a[0, 0, 0, 1] = np.nan
    a[1, 1, 1, 2] = 5.0
    axis = (3,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.zeros((2, 2, 2, 1), dtype=np.float32)
    mean[1, 1, 1, 0] = 5.0 / 3.0
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

    # Input 8: Integer array input with float32 computation
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = (1,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones((2, 3), dtype=bool)
    mean = np.array([[2.0], [5.0]], dtype=np.float32)
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

    # Input 9: 2D array with negative axis indexing (-1,)
    a = np.array([[np.nan, 2.0], [3.0, 4.0], [-1.0, np.nan]], dtype=np.float32)
    axis = (-1,)
    dtype = np.dtype(np.float32)
    ddof = 1
    keepdims = True
    where = np.array([[False, True], [True, True], [True, False]], dtype=bool)
    mean = np.array([[2.0], [3.5], [-1.0]], dtype=np.float32)
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

    # Input 10: 3D array, reduce along (1, 2)
    a = np.array(
        [
            [[1.0, -1.0], [np.nan, 2.0]],
            [[np.nan, np.nan], [3.0, 4.0]],
            [[-2.0, 0.0], [1.0, np.nan]],
        ],
        dtype=np.float32,
    )
    axis = (1, 2)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.array(
        [[[2.0 / 3.0]], [[3.5]], [[-1.0 / 3.0]]], dtype=np.float32
    )
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


generated_inputs["jax.numpy.nanvar_2"] = nanvar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanvar_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanvar_2'.")


check_valid('jax.numpy.nanvar', generated_inputs['jax.numpy.nanvar_2'], lib="jax", suffix=2)
