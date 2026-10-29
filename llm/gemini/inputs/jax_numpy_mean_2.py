
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis (0,), dtype float32, keepdims False
    a = np.array([1.0, -2.5, 3.5, 4.0, -5.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 2: 2D float64 array, axis (1,), dtype float64, keepdims True
    a = np.array(
        [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]],
        dtype=np.float64,
    )
    axis = (1,)
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 3: 2D int32 array, axis (0,), dtype float32, keepdims False
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array(
        [[True, False, True], [True, True, False], [False, True, True]],
        dtype=bool,
    )
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 4: 3D float32 array, multiple axes (0, 2), dtype float32, keepdims True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 5: 3D float64 array, negative axis (-1,), dtype float64, keepdims False
    a = np.linspace(-10.0, 10.0, 24, dtype=np.float64).reshape((2, 3, 4))
    axis = (-1,)
    dtype = np.dtype(np.float64)
    keepdims = False
    where = a > 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 6: 4D float32 array, axis (1, 3), dtype float32, keepdims False
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 4)).astype(np.float32)
    axis = (1, 3)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 3, 2, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 7: 2D complex64 array, axis (0,), dtype complex64, keepdims True
    a = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    axis = (0,)
    dtype = np.dtype(np.complex64)
    keepdims = True
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 8: 1D int64 array with negative elements, axis (-1,), dtype float64, keepdims True
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    axis = (-1,)
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 9: 2D float32 array with broadcast-compatible where (1, 4), axis (0,), dtype float32, keepdims False
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.array([[True, False, True, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    # Input 10: 4D float64 array, all axes reduced (0, 1, 2, 3), dtype float64, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    axis = (0, 1, 2, 3)
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.mean_2"] = mean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mean_2'.")


check_valid('jax.numpy.mean', generated_inputs['jax.numpy.mean_2'], lib="jax", suffix=2)
