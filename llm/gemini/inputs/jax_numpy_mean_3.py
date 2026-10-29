
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def mean_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reduce axis [0], float32
    a = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 2: 2D array, reduce axis [1], keepdims True, float64
    a = np.random.randn(3, 4).astype(np.float64)
    axis = [1]
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 3: 3D array, reduce multiple axes [0, 2], float32
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = [0, 2]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 4: 2D integer array, reduce axis [-1], keepdims True, target dtype float64
    a = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)
    axis = [-1]
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.array(
        [[True, False, True, True], [False, True, True, True], [True, True, False, True]],
        dtype=bool,
    )
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 5: 4D array, reduce axes [1, 3], keepdims False, float32
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = [1, 3]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 3, 4, 5), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 6: 2D array, reduce axis [0], keepdims True, float64
    a = np.linspace(-10.0, 10.0, 12, dtype=np.float64).reshape(4, 3)
    axis = [0]
    dtype = np.dtype(np.float64)
    keepdims = True
    where = a > 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 7: 3D array, negative axis [-2], keepdims False, float32
    a = np.random.uniform(-5.0, 5.0, size=(4, 5, 2)).astype(np.float32)
    axis = [-2]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((4, 5, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 8: 1D complex array, axis [-1], keepdims True, complex64
    a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    axis = [-1]
    dtype = np.dtype(np.complex64)
    keepdims = True
    where = np.array([True, True, False, True, False, True, True, False], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 9: 2D array, reduce all axes [0, 1], keepdims False, float32
    a = np.random.randn(5, 5).astype(np.float32)
    axis = [0, 1]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 10: 3D array with broadcastable where mask, axis [1], keepdims True, float64
    a = np.random.randn(2, 4, 3).astype(np.float64)
    axis = [1]
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.array([[[True], [True], [False], [True]]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 11: 4D array, negative axes [-3, -1], keepdims True, float32
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = [-3, -1]
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.ones((2, 2, 3, 3), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": copy.deepcopy(axis),
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.mean_3"] = mean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mean_3'.")


check_valid('jax.numpy.mean', generated_inputs['jax.numpy.mean_3'], lib="jax", suffix=3)
