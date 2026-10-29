
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmin_inputs():
    list_of_inputs = []

    # Case 1: 1D array, axis=0, keepdims=False
    a = np.array([1.0, np.nan, 3.0, -2.0, np.nan], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(100.0, dtype=np.float32)
    where = np.array([True, True, True, True, False], dtype=bool)
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

    # Case 2: 2D array, axis=1, keepdims=True
    a = np.array(
        [[np.nan, 2.0, 3.0], [-1.0, np.nan, -5.0]], dtype=np.float32
    )
    axis = 1
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
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

    # Case 3: 2D array, axis=0, keepdims=False, broadcasted where
    a = np.array([[1.5, -2.5], [np.nan, 4.0], [-10.0, np.nan]], dtype=np.float64)
    axis = 0
    keepdims = False
    initial = np.array(np.inf, dtype=np.float64)
    where = np.array([[True, False]], dtype=bool)
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

    # Case 4: 3D array, axis=-1, keepdims=True
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 3] = np.nan
    axis = -1
    keepdims = True
    initial = np.array(50.0, dtype=np.float32)
    where = a > 5.0
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

    # Case 5: 3D array, axis=1, negative axis indexing
    a = np.array(
        [
            [[np.nan, 1.0], [2.0, np.nan]],
            [[3.0, 4.0], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    axis = 1
    keepdims = False
    initial = np.array(1000.0, dtype=np.float32)
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

    # Case 6: 4D array, axis=2, float64
    rng = np.random.default_rng(42)
    a = rng.normal(size=(2, 2, 3, 2)).astype(np.float64)
    a[0, 0, 0, 0] = np.nan
    axis = 2
    keepdims = True
    initial = np.array(10.0, dtype=np.float64)
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

    # Case 7: All NaNs in some slices, axis=0
    a = np.array([[np.nan, np.nan], [np.nan, 1.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(5.0, dtype=np.float32)
    where = np.array([[True, True], [True, False]], dtype=bool)
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

    # Case 8: 1D array with negative numbers, axis=-1
    a = np.array([-10.0, -20.0, np.nan, -5.0], dtype=np.float32)
    axis = -1
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, True, False], dtype=bool)
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

    # Case 9: 2D array, axis=-2, initial array matching broadcast
    a = np.array([[np.nan, 3.0], [4.0, 5.0]], dtype=np.float32)
    axis = -2
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, True], [False, True]], dtype=bool)
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

    # Case 10: 3D array with all elements evaluated, axis=0
    a = np.array([[[np.nan]], [[2.5]], [[-1.5]]], dtype=np.float64)
    axis = 0
    keepdims = True
    initial = np.array(np.inf, dtype=np.float64)
    where = np.array([[[True]], [[True]], [[True]]], dtype=bool)
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


generated_inputs["jax.numpy.nanmin_1"] = nanmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmin_1'.")


check_valid('jax.numpy.nanmin', generated_inputs['jax.numpy.nanmin_1'], lib="jax", suffix=1)
