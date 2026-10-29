
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmin_inputs():
    list_of_inputs = []

    # Input 1: 2D array, reduce along axis (0,), keepdims=False
    a = np.array([[1.0, np.nan, 3.0], [-2.0, 4.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 2: 2D array, reduce along axis (1,), keepdims=True
    a = np.array(
        [[np.nan, -5.0, 2.0], [3.0, np.nan, -10.0]], dtype=np.float64
    )
    axis = (1,)
    keepdims = True
    initial = np.array(100.0, dtype=np.float64)
    where = np.array([[True, True, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 3: 1D array, reduce along axis (0,), keepdims=False
    a = np.array([-1.5, np.nan, 2.5, -3.5, np.nan], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 4: 3D array, reduce along axes (0, 2), keepdims=False
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    axis = (0, 2)
    keepdims = False
    initial = np.array(50.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    where[0, 0, 0] = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 5: 3D array, reduce along axis (-1,), keepdims=True, broadcasted where
    a = np.array(
        [[[np.nan, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [np.nan, np.nan]]],
        dtype=np.float64,
    )
    axis = (-1,)
    keepdims = True
    initial = np.array(np.inf, dtype=np.float64)
    where = np.array([[[True, False]], [[False, True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 6: 4D array, reduce along axes (1, 2), keepdims=False
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    axis = (1, 2)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 2, 3, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 7: 2D array, all NaNs with initial value fallback
    a = np.full((3, 3), np.nan, dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 8: 3D array with negative numbers, reduce along axis (1,), keepdims=False
    a = np.array(
        [
            [[-10.0, -20.0], [np.nan, -40.0]],
            [[-50.0, np.nan], [-70.0, -80.0]],
        ],
        dtype=np.float32,
    )
    axis = (1,)
    keepdims = False
    initial = np.array(-5.0, dtype=np.float32)
    where = np.array([[[True, True]], [[True, False]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 9: 1D array, reduce along axis (-1,), keepdims=True
    a = np.array([np.nan, 42.0, np.nan, -100.0], dtype=np.float64)
    axis = (-1,)
    keepdims = True
    initial = np.array(500.0, dtype=np.float64)
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 10: 4D array, reduce along axes (0, 1, 2, 3), keepdims=True
    a = np.array([[[[np.nan, 1.0], [2.0, 3.0]]]], dtype=np.float32)
    axis = (0, 1, 2, 3)
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmin_3"] = nanmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmin_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmin_3'.")


check_valid('jax.numpy.nanmin', generated_inputs['jax.numpy.nanmin_3'], lib="jax", suffix=3)
