
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanargmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs and negative values, axis=0, keepdims=False
    a = np.array([1.0, -3.0, np.nan, 5.0, -2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, axis=0, keepdims=False
    a = np.array([[1.5, np.nan], [2.0, 3.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64, axis=1, keepdims=True
    a = np.array([[np.nan, 4.5, -1.0], [3.0, np.nan, 2.0]], dtype=np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=2, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with an all-NaN row to check the -1 return behavior, axis=1, keepdims=False
    a = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, negative axis=-1, keepdims=False
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 0, 0, 0] = np.nan
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with only NaNs, axis=0, keepdims=True
    a = np.full((3, 3), np.nan, dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=-2, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {"a": a, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, float64, axis=0, keepdims=True
    a = np.array([np.nan, -10.0, np.nan, -5.0], dtype=np.float64)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D array, axis=1, keepdims=True
    a = np.random.randn(10, 15).astype(np.float32)
    a[a < 0] = np.nan
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanargmax"] = nanargmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanargmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanargmax'.")


check_valid('jax.numpy.nanargmax', generated_inputs['jax.numpy.nanargmax'], lib="jax", suffix=0)
