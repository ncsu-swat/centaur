
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanmin_inputs():
    list_of_inputs = []
    
    # Input 1
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, True, False], [True, False, True]], dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})
    
    # Input 2
    a = np.array([[np.nan, -2.0], [3.0, -4.0], [5.0, np.nan]], dtype=np.float64)
    axis = (1,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float64)
    where = np.array([[True, True], [False, True], [True, False]], dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 3
    a = np.array([1.5, -3.2, np.nan, 4.8, -0.5], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(2.0, dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 4
    a = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    a[0, 1, 1] = np.nan
    axis = (1, 2)
    keepdims = True
    initial = np.array(15.0, dtype=np.float32)
    where = np.ones((2, 2, 3), dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 5
    a = np.array([[[[1.0, 2.0], [np.nan, 4.0]]]], dtype=np.float32)
    axis = (2,)
    keepdims = False
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((1, 1, 2, 2), dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 6
    a = np.array([[-10.0, -20.0], [-30.0, -40.0]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(-5.0, dtype=np.float32)
    where = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 7
    a = np.random.randn(5, 5).astype(np.float64)
    a[a < 0] = np.nan
    axis = (1,)
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = np.random.choice([True, False], size=(5, 5))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 8
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = (0, 2)
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((3, 3, 3), dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 9
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, True, True], dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    # Input 10
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    axis = (0, 1, 2)
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = np.ones((2, 3, 4), dtype=np.bool_)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.nanmin_3"] = nanmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmin_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmin_3'.")


check_valid('jax.numpy.nanmin', generated_inputs['jax.numpy.nanmin_3'], lib="jax", suffix=3)
