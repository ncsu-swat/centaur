
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanprod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs, reduce along axis 0
    a = np.array([1.5, np.nan, 2.0, -3.0, np.nan, 4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype("float32")
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, True, True, True, True, True], dtype=bool)
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

    # Input 2: 2D float64 array with NaNs, reduce along axis 1 with keepdims=True
    a = np.array(
        [[np.nan, 2.0, -1.5], [3.0, np.nan, np.nan], [-2.0, -4.0, 0.5]],
        dtype=np.float64,
    )
    axis = (1,)
    dtype = np.dtype("float64")
    keepdims = True
    initial = np.array(-1.0, dtype=np.float64)
    where = np.array(
        [[True, True, False], [True, False, True], [False, True, True]],
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

    # Input 3: 2D float32 array, reduce along axis 0
    a = np.array(
        [[1.0, np.nan, 3.0], [np.nan, 5.0, -2.0], [4.0, 2.0, np.nan]],
        dtype=np.float32,
    )
    axis = (0,)
    dtype = np.dtype("float32")
    keepdims = False
    initial = np.array(2.0, dtype=np.float32)
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

    # Input 4: 3D float32 array, multi-axis reduction (0, 2) with keepdims=True
    a = np.array(
        [
            [[np.nan, 1.0], [2.0, -1.0]],
            [[3.0, np.nan], [np.nan, 4.0]],
            [[-2.0, 5.0], [1.0, np.nan]],
        ],
        dtype=np.float32,
    )
    axis = (0, 2)
    dtype = np.dtype("float32")
    keepdims = True
    initial = np.array(0.5, dtype=np.float32)
    where = np.array(
        [
            [[True, False], [True, True]],
            [[False, True], [True, True]],
            [[True, True], [False, True]],
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

    # Input 5: 3D float64 array, negative axis reduction (-1,)
    a = np.random.randn(2, 4, 3).astype(np.float64)
    a[0, 1, 2] = np.nan
    a[1, 3, 0] = np.nan
    axis = (-1,)
    dtype = np.dtype("float64")
    keepdims = False
    initial = np.array(1.0, dtype=np.float64)
    where = np.ones((2, 4, 3), dtype=bool)
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

    # Input 6: 4D float32 array, multi-axis reduction (1, 3)
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 4)).astype(np.float32)
    a[a < 0.0] = np.nan
    axis = (1, 3)
    dtype = np.dtype("float32")
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(2, 3, 2, 4))
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

    # Input 7: 2D array with all NaNs
    a = np.full((3, 3), np.nan, dtype=np.float32)
    axis = (0,)
    dtype = np.dtype("float32")
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
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

    # Input 8: 1D float64 array, negative values, keepdims=True
    a = np.array([-2.5, -4.0, np.nan, -0.5, 3.0], dtype=np.float64)
    axis = (-1,)
    dtype = np.dtype("float64")
    keepdims = True
    initial = np.array(3.0, dtype=np.float64)
    where = np.array([True, True, True, False, True], dtype=bool)
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

    # Input 9: 3D float32 array, reduction along axis 1, output converted to float64
    a = np.array(
        [
            [[1.0, 2.0], [np.nan, 3.0]],
            [[np.nan, np.nan], [4.0, 5.0]],
            [[-1.0, -2.0], [np.nan, 1.0]],
        ],
        dtype=np.float32,
    )
    axis = (1,)
    dtype = np.dtype("float64")
    keepdims = False
    initial = np.array(1.0, dtype=np.float64)
    where = np.ones((3, 2, 2), dtype=bool)
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

    # Input 10: 4D float64 array, multi-axis reduction (0, 1, 2) with keepdims=True
    a = np.random.randn(2, 2, 2, 3).astype(np.float64)
    a[0, 0, 0, 0] = np.nan
    axis = (0, 1, 2)
    dtype = np.dtype("float64")
    keepdims = True
    initial = np.array(-2.0, dtype=np.float64)
    where = np.ones((2, 2, 2, 3), dtype=bool)
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

    # Input 11: 2D array, where is completely False
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = (0, 1)
    dtype = np.dtype("float32")
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.zeros((2, 2), dtype=bool)
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


generated_inputs["jax.numpy.nanprod_2"] = nanprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanprod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanprod_2'.")


check_valid('jax.numpy.nanprod', generated_inputs['jax.numpy.nanprod_2'], lib="jax", suffix=2)
