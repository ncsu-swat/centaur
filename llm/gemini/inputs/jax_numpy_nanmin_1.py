
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmin_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, float32, axis=0, no keepdims
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, keepdims=True
    a = np.array([[1.0, np.nan, 3.0], [4.0, -5.0, np.nan]], dtype=np.float32)
    axis = 1
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones_like(a, dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64, axis=2
    a = np.random.randn(2, 3, 4).astype(np.float64)
    a[0, 1, 2] = np.nan
    axis = 2
    keepdims = False
    initial = np.array(5.0, dtype=np.float64)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, keepdims=True
    a = np.array([np.nan, 2.0, np.nan, -1.0, 5.0], dtype=np.float32)
    axis = 0
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with all NaNs
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    axis = -1
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, float32, axis=2, keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[1, 0, 1, 0] = np.nan
    axis = 2
    keepdims = True
    initial = np.array(2.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, negative values, float32, axis=0
    a = np.array([[-10.0, np.nan, -20.0], [np.nan, -5.0, -1.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 3), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, float32, axis=-1, keepdims=True
    a = np.array([1.5, 2.5, np.nan], dtype=np.float32)
    axis = -1
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.array([True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, float32, axis=1
    a = np.random.randn(2, 2, 3).astype(np.float32)
    axis = 1
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 2, 3), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, broadcastable where mask
    a = np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 6.0]], dtype=np.float32)
    axis = 1
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True], [True], [True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
