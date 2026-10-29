
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanprod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs, axis 0
    a = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, True, True, False], dtype=bool)
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

    # Input 2: 2D float64 array with NaNs, axis 1, keepdims True
    a = np.array([[1.0, np.nan, -2.0], [np.nan, 4.0, 5.0]], dtype=np.float64)
    axis = 1
    dtype = np.dtype(np.float64)
    keepdims = True
    initial = np.array(2.0, dtype=np.float64)
    where = np.ones((2, 3), dtype=bool)
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

    # Input 3: 2D float32 array, negative axis -1
    a = np.array(
        [[-1.5, 2.5], [np.nan, -3.0], [4.0, np.nan]], dtype=np.float32
    )
    axis = -1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array(
        [[True, True], [False, True], [True, False]], dtype=bool
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

    # Input 4: 3D float32 array, axis 0, keepdims True
    a = np.array(
        [
            [[np.nan, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
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

    # Input 5: 3D float64 array with all NaNs along axis 2
    a = np.full((2, 3, 4), np.nan, dtype=np.float64)
    axis = 2
    dtype = np.dtype(np.float64)
    keepdims = False
    initial = np.array(1.0, dtype=np.float64)
    where = np.ones((2, 3, 4), dtype=bool)
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

    # Input 6: 2D float32 array with negative axis -2
    a = np.array([[2.0, np.nan, -1.0], [3.0, 4.0, np.nan]], dtype=np.float32)
    axis = -2
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.5, dtype=np.float32)
    where = np.array([[True, False, True], [True, True, False]], dtype=bool)
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

    # Input 7: 1D array with int32 output dtype
    a = np.array([2.0, 3.0, np.nan, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.array([True, True, True, True], dtype=bool)
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

    # Input 8: 4D float32 array, axis 1
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    a[0, 1, 0, 1] = np.nan
    a[1, 2, 1, 0] = np.nan
    axis = 1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = a > 0
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

    # Input 9: 2D array with broadcastable 'where' tensor
    a = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, False]], dtype=bool)
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

    # Input 10: 1D float64 array, all where elements False
    a = np.array([10.0, 20.0, np.nan, -5.0], dtype=np.float64)
    axis = 0
    dtype = np.dtype(np.float64)
    keepdims = False
    initial = np.array(3.0, dtype=np.float64)
    where = np.zeros(4, dtype=bool)
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


generated_inputs["jax.numpy.nanprod_1"] = nanprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanprod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanprod_1'.")


check_valid('jax.numpy.nanprod', generated_inputs['jax.numpy.nanprod_1'], lib="jax", suffix=1)
