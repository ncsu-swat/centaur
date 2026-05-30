
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmin_inputs():
    list_of_inputs = []

    # Input 1, valid - 2D float32 array, axis (0,), keepdims=True
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, False, True], [True, True, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid - 2D float64 array, axis (1,), keepdims=False
    a = np.array([[np.nan, 2.0], [3.0, np.nan], [5.0, 6.0]], dtype=np.float64)
    axis = (1,)
    keepdims = False
    initial = np.array(5.0, dtype=np.float64)
    where = np.array([[True, True], [False, True], [True, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid - 1D float32 array, axis (0,), keepdims=True
    a = np.array([np.nan, -1.0, -2.0, np.nan, 5.0], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([False, True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid - 3D float32 array, multi-axis reduction
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    axis = (0, 2)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(2, 3, 4))
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - small 2D array, negative values, keepdims=False
    a = np.array([[-10.0, 20.0], [np.nan, 30.0]], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(15.0, dtype=np.float32)
    where = np.array([[True, True], [True, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - 4D array, axis (1, 3), keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - 2D negative and nan matrix
    a = np.array([[-1.0, -2.0, np.nan], [np.nan, -3.0, -4.0], [-5.0, np.nan, -6.0]], dtype=np.float32)
    axis = (1,)
    keepdims = False
    initial = np.array(-0.5, dtype=np.float32)
    where = np.array([[True, True, False], [False, True, True], [True, False, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - 3D array with broadcasting 'where'
    a = np.random.randn(3, 1, 4).astype(np.float32)
    axis = (2,)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(3, 1, 4))
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - long 1D array
    a = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(100.0, dtype=np.float32)
    where = np.array([True] * 10, dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - 2D float64 with reduction over all axes
    a = np.random.randn(5, 5).astype(np.float64)
    axis = (0, 1)
    keepdims = True
    initial = np.array(2.0, dtype=np.float64)
    where = np.random.choice([True, False], size=(5, 5))
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
