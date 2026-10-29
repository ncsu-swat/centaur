
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nansum_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, axis (0,)
    a = np.array([1.0, np.nan, 3.0, -4.0, np.nan], dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0,),
            "dtype": np.float32,
            "keepdims": False,
            "initial": 0.0,
            "where": where,
        }
    )

    # Input 2: 2D array, reduction along axis (1,) with keepdims=True
    a = np.array(
        [[np.nan, 2.5, -1.0], [4.0, np.nan, np.nan]], dtype=np.float64
    )
    where = np.ones(a.shape, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (1,),
            "dtype": np.float64,
            "keepdims": True,
            "initial": 1.5,
            "where": where,
        }
    )

    # Input 3: 2D array, reduction along axis (0,)
    a = np.array([[-3.0, 7.2], [np.nan, -5.1], [0.0, np.nan]], dtype=np.float32)
    where = np.array([[True, False], [True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0,),
            "dtype": np.float32,
            "keepdims": False,
            "initial": -2.0,
            "where": where,
        }
    )

    # Input 4: 2D array, reduce all axes (0, 1)
    a = np.array(
        [[10.0, np.nan], [np.nan, -20.0], [5.0, 2.0]], dtype=np.float32
    )
    where = np.ones(a.shape, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0, 1),
            "dtype": np.float32,
            "keepdims": False,
            "initial": 100.0,
            "where": where,
        }
    )

    # Input 5: 3D array, negative axis (-1,)
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = a > 5.0
    list_of_inputs.append(
        {
            "a": a,
            "axis": (-1,),
            "dtype": np.float64,
            "keepdims": True,
            "initial": 0.0,
            "where": where,
        }
    )

    # Input 6: 3D array, reduce multiple axes (0, 2)
    a = np.full((3, 2, 4), np.nan, dtype=np.float32)
    a[0, 0, 1] = 4.5
    a[2, 1, 3] = -8.0
    where = np.ones((3, 2, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0, 2),
            "dtype": np.float32,
            "keepdims": False,
            "initial": 5.0,
            "where": where,
        }
    )

    # Input 7: 4D array, axis (1, 3) with keepdims=True
    a = np.random.randn(2, 3, 2, 2).astype(np.float32)
    a[0, 1, :, :] = np.nan
    where = np.ones((2, 3, 2, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (1, 3),
            "dtype": np.float32,
            "keepdims": True,
            "initial": -0.5,
            "where": where,
        }
    )

    # Input 8: All NaNs in 2D array
    a = np.full((4, 4), np.nan, dtype=np.float32)
    where = np.ones((4, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0,),
            "dtype": np.float32,
            "keepdims": False,
            "initial": 0.0,
            "where": where,
        }
    )

    # Input 9: Broadcasted where mask with 3D array
    a = np.array(
        [[[1.0, 2.0], [np.nan, 4.0]], [[5.0, -6.0], [7.0, np.nan]]],
        dtype=np.float64,
    )
    where = np.array([[[True, False]], [[False, True]]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (1,),
            "dtype": np.float64,
            "keepdims": False,
            "initial": 10.0,
            "where": where,
        }
    )

    # Input 10: 1D array with negative values and negative initial
    a = np.array([-10.5, -20.5, np.nan, -30.0], dtype=np.float32)
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": (0,),
            "dtype": np.float32,
            "keepdims": True,
            "initial": -5.0,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nansum_3"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nansum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nansum_3'.")


check_valid('jax.numpy.nansum', generated_inputs['jax.numpy.nansum_3'], lib="jax", suffix=3)
