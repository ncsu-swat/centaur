
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, single axis tuple
    a = np.array([1.0, -2.5, 3.5, 4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 2: 2D float64 array, reduce along axis 1 with keepdims
    a = np.array([[-1.5, 2.0, 3.5], [4.2, -5.1, 0.0]], dtype=np.float64)
    axis = (1,)
    dtype = np.dtype(np.float64)
    keepdims = True
    initial = np.array(1.0, dtype=np.float64)
    where = np.ones((2, 3), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 3: 2D int32 array, reduce all axes
    a = np.array([[10, -5, 2], [8, 4, -3]], dtype=np.int32)
    axis = (0, 1)
    dtype = np.dtype(np.int64)
    keepdims = False
    initial = np.array(5, dtype=np.int64)
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 4: 3D float32 array, negative axis index
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4)) - 10.0
    axis = (-1,)
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(-2.0, dtype=np.float32)
    where = (a > -5.0)
    promote_integers = False
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 5: 3D int16 array, multiple axes reduction
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    axis = (0, 2)
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.ones((2, 2, 2), dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 6: Broadcasted where condition on 2D array
    a = np.array([[2.5, -1.0, 4.0], [3.0, 5.5, -2.0]], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, False, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 7: Empty tuple for axis (sum over no axes, identity reduction)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = ()
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 8: Complex numbers
    a = np.array([[1.0 + 2.0j, -3.0 + 0.5j], [0.0 - 1.0j, 4.0 - 2.0j]], dtype=np.complex64)
    axis = (1,)
    dtype = np.dtype(np.complex64)
    keepdims = True
    initial = np.array(1.0 + 0.0j, dtype=np.complex64)
    where = np.array([[True, False], [True, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 9: 4D float32 array, reduction along intermediate axes
    a = np.random.uniform(-1.0, 1.0, size=(2, 3, 2, 4)).astype(np.float32)
    axis = (1, 2)
    dtype = np.dtype(np.float64)
    keepdims = False
    initial = np.array(0.0, dtype=np.float64)
    where = np.ones((2, 3, 2, 4), dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    # Input 10: Boolean input array with integer sum
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    axis = (0,)
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([[True, True, True], [True, False, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    })

    return list_of_inputs

generated_inputs["jax.numpy.sum_2"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_2'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_2'], lib="jax", suffix=2)
