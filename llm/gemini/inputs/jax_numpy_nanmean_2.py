
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmean_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, axis=(0,)
    a = np.array([1.0, np.nan, 3.0, -4.0, 5.0], dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with NaNs, axis=(1,), keepdims=True
    a = np.array([[np.nan, 2.0, 3.0], [4.0, -5.0, np.nan]], dtype=np.float64)
    where = np.array([[True, True, False], [True, True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype(np.float64),
        "keepdims": True,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with broadcastable 'where', axis=(0,)
    a = np.array(
        [[-10.0, 20.0], [np.nan, -40.0], [50.0, 60.0]], dtype=np.float32
    )
    where = np.array([[True, True], [False, True], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=(0, 2), keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    where = np.ones((2, 3, 4), dtype=bool)
    where[0, 0, 0] = False
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "dtype": np.dtype(np.float32),
        "keepdims": True,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative axis=(-1,), dtype=float64
    a = np.array(
        [[[np.nan, 1.5], [-2.5, np.nan]], [[3.0, 4.0], [np.nan, np.nan]]],
        dtype=np.float64,
    )
    where = np.array(
        [[[True, True], [True, False]], [[False, True], [True, True]]],
        dtype=bool,
    )
    input_dict = {
        "a": a,
        "axis": (-1,),
        "dtype": np.dtype(np.float64),
        "keepdims": False,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=(1, 3), keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, :, 1, :] = np.nan
    where = np.ones((2, 2, 2, 2), dtype=bool)
    where[1, 0, 0, 1] = False
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "dtype": np.dtype(np.float32),
        "keepdims": True,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, axis=(0, 1) to reduce all axes
    a = np.array([[np.nan, np.nan], [-1.0, 2.0]], dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative values and float64 output
    a = np.array([-100.5, -200.25, np.nan, -0.125], dtype=np.float64)
    where = np.array([True, True, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "dtype": np.dtype(np.float64),
        "keepdims": True,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array reducing two negative axes, axis=(-2, -1)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    a[a < 0] = np.nan
    where = np.ones((3, 4, 5), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (-2, -1),
        "dtype": np.dtype(np.float32),
        "keepdims": False,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with 1D broadcastable where mask
    a = np.array(
        [[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0], [7.0, -8.0, 9.0]],
        dtype=np.float64,
    )
    where = np.array([[True], [False], [True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "dtype": np.dtype(np.float64),
        "keepdims": True,
        "where": where,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanmean_2"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmean_2'.")


check_valid('jax.numpy.nanmean', generated_inputs['jax.numpy.nanmean_2'], lib="jax", suffix=2)
