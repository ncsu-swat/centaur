
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmedian_inputs():
    list_of_inputs = []

    # Input 1: 2D array with some NaNs, float32, axis=0, keepdims=False
    a = np.array([[2.0, np.nan, 7.0, np.nan],
                  [np.nan, 5.0, 9.0, 2.0],
                  [6.0, 1.0, np.nan, 3.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with NaNs, float64, axis=1, keepdims=True
    a = np.array([[1.5, np.nan, 3.2],
                  [np.nan, np.nan, 4.5],
                  [0.5, 9.1, np.nan]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array with NaNs, axis=0, keepdims=False, overwrite_input=False
    a = np.array([np.nan, 3.0, np.nan, 1.0, 5.0, np.nan], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with NaNs, float32, axis=2, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[a > 1.0] = np.nan
    input_dict = {
        "a": a,
        "axis": 2,
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with all NaNs, axis=0
    a = np.full((3, 3), np.nan, dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with negative values and NaNs, float64, negative axis=-1
    a = np.array([[-10.0, np.nan, -2.5],
                  [np.nan, -5.0, -1.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": -1,
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32, axis=1, keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 1, 0, 1] = np.nan
    a[1, 0, 1, 0] = np.nan
    input_dict = {
        "a": a,
        "axis": 1,
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with negative values, float32, axis=-2, overwrite_input=False
    a = np.array([[[-1.0, np.nan], [2.0, -3.0]],
                  [[np.nan, 4.0], [-5.0, np.nan]]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "keepdims": False,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with no NaNs, axis=-1, keepdims=True
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with NaNs and infinities, float64, axis=0, keepdims=True
    a = np.array([[np.inf, np.nan, -np.inf],
                  [1.0, 2.0, 3.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "keepdims": True,
        "overwrite_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanmedian_1"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmedian_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmedian_1'.")


check_valid('jax.numpy.nanmedian', generated_inputs['jax.numpy.nanmedian_1'], lib="jax", suffix=1)
