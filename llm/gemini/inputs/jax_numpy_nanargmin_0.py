
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanargmin_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, float32, keepdims=False
    a = np.array([np.nan, -3.0, 5.0, 4.0, 2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with NaNs, float64, keepdims=True
    a = np.array([[1.0, 3.0, np.nan], [5.0, 4.0, np.nan]], dtype=np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, no NaNs, axis=0, keepdims=False
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float64, all NaNs, axis=2, keepdims=True
    a = np.full((2, 3, 4), np.nan, dtype=np.float64)
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, negative axis, keepdims=False
    a = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, float64, keepdims=True
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with positive values and NaNs, keepdims=True
    a = np.array([10.0, np.nan, 2.0, np.nan, 8.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, a column of NaNs, axis=0, keepdims=False
    a = np.array([[np.nan, 2.0], [np.nan, 1.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D array, axis=1, keepdims=True
    a = np.random.uniform(-100, 100, (10, 20)).astype(np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, axis=1, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanargmin"] = nanargmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanargmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanargmin'.")


check_valid('jax.numpy.nanargmin', generated_inputs['jax.numpy.nanargmin'], lib="jax", suffix=0)
