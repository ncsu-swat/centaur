
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmin_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, reduce along axis 0
    a = np.array([[1.0, np.nan, 3.0], [-2.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, True, True], [True, False, True]], dtype=bool)
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

    # Input 2: 2D float32 array, reduce along axis 1 with keepdims=True
    a = np.array(
        [[np.nan, -10.0, 4.0], [2.0, np.nan, -5.0], [0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    axis = (1,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[False, True, True], [True, True, False], [True, True, True]], dtype=bool)
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

    # Input 3: 1D float64 array, reduce along axis 0
    a = np.array([np.nan, -3.5, 4.2, np.nan, 1.1, -8.9], dtype=np.float64)
    axis = (0,)
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = np.array([True, True, True, False, True, False], dtype=bool)
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

    # Input 4: 3D float32 array, reduce along axes (0, 2)
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 0, 1] = np.nan
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

    # Input 5: 3D float64 array, broadcastable where mask
    a = np.array(
        [[[1.0, -2.0], [np.nan, 4.0]], [[-5.0, np.nan], [7.0, -8.0]]],
        dtype=np.float64,
    )
    axis = (1,)
    keepdims = True
    initial = np.array(1000.0, dtype=np.float64)
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

    # Input 6: 4D float32 array, multi-axis reduction
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    a[a < -0.5] = np.nan
    axis = (1, 3)
    keepdims = False
    initial = np.array(5.0, dtype=np.float32)
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

    # Input 7: 2D float32 array, negative axis index
    a = np.array([[-1.0, -2.0, -3.0], [np.nan, np.nan, -0.5]], dtype=np.float32)
    axis = (-1,)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, True, False], [True, True, True]], dtype=bool)
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

    # Input 8: 1D float32 array, keepdims=True
    a = np.array([np.nan, np.nan, -15.0, 20.0], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, False, True, True], dtype=bool)
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

    # Input 9: 3D float32 array, reduction over all axes
    a = np.array([[[np.nan, 2.0], [3.0, -4.0]]], dtype=np.float32)
    axis = (0, 1, 2)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[[True, True], [False, True]]], dtype=bool)
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

    # Input 10: 2D float64 array, reduce across axis (0, 1) with keepdims=False
    a = np.array(
        [[10.5, -20.3, np.nan], [np.nan, 0.0, -5.5]], dtype=np.float64
    )
    axis = (0, 1)
    keepdims = False
    initial = np.array(np.inf, dtype=np.float64)
    where = np.array([[True, True, True], [True, False, True]], dtype=bool)
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


generated_inputs["jax.numpy.nanmin_2"] = nanmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmin_2'.")


check_valid('jax.numpy.nanmin', generated_inputs['jax.numpy.nanmin_2'], lib="jax", suffix=2)
