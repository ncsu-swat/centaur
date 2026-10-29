
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def all_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, axis=0, keepdims=False
    a = np.array([True, True, False, True], dtype=bool)
    axis = 0
    keepdims = False
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 2: 2D boolean array, axis=0, keepdims=True
    a = np.array([[True, False], [True, True]], dtype=bool)
    axis = 0
    keepdims = True
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 3: 2D float array with negative and positive values, axis=1, keepdims=False
    a = np.array([[-1.5, 0.0, 2.3], [1.0, -2.0, 3.5]], dtype=np.float32)
    axis = 1
    keepdims = False
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 4: 3D integer array, axis=2, keepdims=True
    a = np.arange(-6, 6, dtype=np.int32).reshape(2, 3, 2)
    axis = 2
    keepdims = True
    where = np.ones((2, 3, 2), dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 5: 3D boolean array, negative axis=-1, broadcasted where
    a = np.zeros((2, 4, 3), dtype=bool)
    axis = -1
    keepdims = False
    where = np.array([[[True], [False], [True], [True]]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 6: 1D int64 array with zero, axis=-1, keepdims=True
    a = np.array([1, 2, 0, -4, 5], dtype=np.int64)
    axis = -1
    keepdims = True
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 7: 4D float64 array, axis=1, keepdims=False
    a = np.ones((2, 2, 2, 2), dtype=np.float64)
    axis = 1
    keepdims = False
    where = np.zeros((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 8: 2D complex array, negative axis=-2, keepdims=True
    a = np.array([[1.0 + 2.0j, 0.0 + 0.0j], [-1.0 - 1.0j, 3.0 + 0.0j]], dtype=np.complex64)
    axis = -2
    keepdims = True
    where = np.array([[True, False]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 9: 3D boolean array, axis=0, keepdims=False
    a = np.ones((3, 2, 4), dtype=bool)
    axis = 0
    keepdims = False
    where = np.array([[[True], [False]]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 10: 2D int8 array with negative numbers, axis=1, keepdims=True
    a = np.array([[-128, 127], [0, -1]], dtype=np.int8)
    axis = 1
    keepdims = True
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    return list_of_inputs


generated_inputs["jax.numpy.all_1"] = all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_1'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_1'], lib="jax", suffix=1)
