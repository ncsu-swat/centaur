
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs
    a = np.array([1.0, np.nan, 3.5, -2.0, np.nan, 4.0], dtype=np.float32)
    where = np.array([True, True, True, False, True, True], dtype=bool)
    initial = np.array(-100.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 0,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 2: 2D float32 array, axis=1, keepdims=True
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, np.nan, -1.0], [-5.0, 0.0, np.nan]],
        dtype=np.float32,
    )
    where = np.ones((3, 3), dtype=bool)
    initial = np.array(-10.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 1,
            "keepdims": True,
            "initial": initial,
            "where": where,
        }
    )

    # Input 3: 2D float64 array with negative axis, keepdims=False
    a = np.array(
        [[-10.5, np.nan, -2.3], [-0.5, -1.2, np.nan]], dtype=np.float64
    )
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    initial = np.array(-50.0, dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "axis": -1,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 4: 3D float32 array, axis=0, keepdims=True
    np.random.seed(42)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = np.ones((2, 3, 4), dtype=bool)
    where[0, 0, 0] = False
    initial = np.array(-1e5, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 0,
            "keepdims": True,
            "initial": initial,
            "where": where,
        }
    )

    # Input 5: 3D float64 array, broadcastable where mask along axis 1
    a = np.random.randn(4, 5, 3).astype(np.float64)
    a[a < -0.5] = np.nan
    where = np.array([[[True], [False], [True], [True], [False]]], dtype=bool)
    where = np.broadcast_to(where, (4, 5, 1))
    initial = np.array(-999.0, dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 1,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 6: 4D float32 array, axis=-2, keepdims=True
    a = np.random.uniform(-10.0, 10.0, size=(2, 3, 4, 2)).astype(np.float32)
    a[0, :, 1, :] = np.nan
    where = np.random.choice([True, False], size=(2, 3, 4, 2), p=[0.8, 0.2])
    initial = np.array(-20.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": -2,
            "keepdims": True,
            "initial": initial,
            "where": where,
        }
    )

    # Input 7: 1D float64 array with all NaNs and false where mask
    a = np.full((10,), np.nan, dtype=np.float64)
    where = np.zeros((10,), dtype=bool)
    initial = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 0,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 8: 2D float32 array, axis=0, keepdims=False with large values
    a = np.array(
        [[1e4, np.nan, 3e4], [4e4, 5e4, np.nan], [np.nan, 2e4, 1e4]],
        dtype=np.float32,
    )
    where = np.array([[True, True, True], [True, True, True], [True, True, True]], dtype=bool)
    initial = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 0,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 9: 5D float32 array with negative axis
    a = np.random.randn(2, 2, 2, 3, 2).astype(np.float32)
    a[1, 0, 1, :, :] = np.nan
    where = np.ones((2, 2, 2, 3, 2), dtype=bool)
    initial = np.array(-1000.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": -3,
            "keepdims": False,
            "initial": initial,
            "where": where,
        }
    )

    # Input 10: 3D float32 array, axis=2, keepdims=True with 1D broadcastable where
    a = np.array(
        [[[np.nan, 1.0], [2.0, 3.0]], [[4.0, np.nan], [np.nan, 7.0]]],
        dtype=np.float32,
    )
    where = np.array([[[True, False]]], dtype=bool)
    initial = np.array(-5.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "axis": 2,
            "keepdims": True,
            "initial": initial,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmax_1"] = nanmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmax_1'.")


check_valid('jax.numpy.nanmax', generated_inputs['jax.numpy.nanmax_1'], lib="jax", suffix=1)
