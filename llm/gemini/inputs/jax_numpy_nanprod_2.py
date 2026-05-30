
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanprod_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = (0,)
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, True], [True, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float64)
    axis = (1,)
    dtype = np.float64
    keepdims = True
    initial = np.array(2.0, dtype=np.float64)
    where = np.array([[True, True, False], [False, True, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([np.nan, -2.0, 3.0, -4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.float32
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([True, True, True, False], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]], dtype=np.float32)
    axis = (0, 2)
    dtype = np.float32
    keepdims = True
    initial = np.array(2.0, dtype=np.float32)
    where = np.ones((2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    axis = (1,)
    dtype = np.float32
    keepdims = False
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.array([10.0, np.nan, 20.0, 30.0], dtype=np.float64)
    axis = (0,)
    dtype = np.float64
    keepdims = True
    initial = np.array(0.1, dtype=np.float64)
    where = np.array([True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[[np.nan]]], dtype=np.float32)
    axis = (0, 1, 2)
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((1, 1, 1), dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[1.5, -1.5], [np.nan, 2.5]], dtype=np.float32)
    axis = (1,)
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = (0,)
    dtype = np.float64
    keepdims = True
    initial = np.array(10.0, dtype=np.float64)
    where = np.array([[True, True, True], [False, False, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([np.nan, np.nan, 2.0, np.nan], dtype=np.float32)
    axis = (-1,)
    dtype = np.float32
    keepdims = False
    initial = np.array(3.0, dtype=np.float32)
    where = np.array([True, True, True, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanprod_2"] = nanprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanprod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanprod_2'.")


check_valid('jax.numpy.nanprod', generated_inputs['jax.numpy.nanprod_2'], lib="jax", suffix=2)
