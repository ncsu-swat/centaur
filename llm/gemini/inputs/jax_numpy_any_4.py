
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def any_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, single axis reduction
    a = np.array([False, True, False, False], dtype=bool)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    where = np.array([True, True, True, True], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 2: 2D float32 array with negative and positive values, keepdims=True
    a = np.array([[-1.5, 0.0, 2.3], [0.0, -0.0, 0.0]], dtype=np.float32)
    axis = np.array([1], dtype=np.int64)
    keepdims = True
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 3: 2D integer array, reducing along axis 0
    a = np.array([[0, 1, 0], [0, 0, 0], [3, 0, -2]], dtype=np.int32)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    where = np.array([[True, True, False], [False, True, True], [True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 4: 3D float64 array, multi-axis reduction
    a = np.zeros((2, 3, 4), dtype=np.float64)
    a[0, 1, 2] = 1.0
    axis = np.array([0, 2], dtype=np.int32)
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 5: 3D int8 array with negative axis
    a = np.array([[[0, 1], [-1, 0]], [[0, 0], [2, 0]]], dtype=np.int8)
    axis = np.array([-1], dtype=np.int32)
    keepdims = False
    where = np.array([[[True, False], [True, True]], [[False, True], [True, False]]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 6: 4D boolean array, reducing over multiple axes
    a = np.random.choice([True, False], size=(2, 2, 3, 2))
    axis = np.array([1, 3], dtype=np.int64)
    keepdims = False
    where = np.ones((2, 2, 3, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 7: 2D complex64 array with broadcasted 'where'
    a = np.array([[0.0 + 0.0j, 1.0 - 1.0j], [0.0 + 2.0j, 0.0 + 0.0j]], dtype=np.complex64)
    axis = np.array([0], dtype=np.int32)
    keepdims = True
    where = np.array([[True, False]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 8: 1D int64 array with all zeros
    a = np.zeros((10,), dtype=np.int64)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    where = np.zeros((10,), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 9: 3D float32 array with inf and nan values
    a = np.array([[[np.nan, 0.0], [np.inf, -np.inf]], [[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    axis = np.array([1], dtype=np.int32)
    keepdims = False
    where = np.array([[[True, True], [False, True]], [[True, False], [True, True]]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 10: 5D boolean array, single axis reduction with keepdims=True
    a = np.zeros((2, 1, 3, 1, 2), dtype=bool)
    a[1, 0, 2, 0, 1] = True
    axis = np.array([2], dtype=np.int32)
    keepdims = True
    where = np.ones((2, 1, 3, 1, 2), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 11: 2D array reducing over all axes via array([0, 1])
    a = np.array([[False, False], [False, False]], dtype=bool)
    axis = np.array([0, 1], dtype=np.int32)
    keepdims = False
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.any_4"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_4'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_4'], lib="jax", suffix=4)
