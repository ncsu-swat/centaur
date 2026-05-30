
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_mean_inputs():
    list_of_inputs = []

    # Input 1, 1D float32 array, simple mean along axis 0
    a = np.random.randn(5).astype(np.float32)
    axis = (0,)
    dtype = np.dtype('float32')
    keepdims = True
    where = np.ones((5,), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 2, 2D float64 array, mean along axis 1
    a = np.random.randn(3, 4).astype(np.float64)
    axis = (1,)
    dtype = np.dtype('float64')
    keepdims = False
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 3, 3D integer array with a partial where mask, custom output dtype
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    axis = (0, 2)
    dtype = np.dtype('float32')
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 4, 2D float32 array, mean over axis 0, without keeping dims
    a = np.random.randn(5, 5).astype(np.float32)
    axis = (0,)
    dtype = np.dtype('float32')
    keepdims = False
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 5, 4D float16 array, mean along multiple axes with keepdims
    a = np.random.randn(2, 2, 2, 2).astype(np.float16)
    axis = (1, 3)
    dtype = np.dtype('float32')
    keepdims = True
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 6, 1D int64 array, output as float64
    a = np.random.randint(1, 100, size=(10,)).astype(np.int64)
    axis = (0,)
    dtype = np.dtype('float64')
    keepdims = False
    where = np.ones((10,), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 7, 3D float32 array, mean along last axis
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (2,)
    dtype = np.dtype('float32')
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 8, 2D float32 array, mean over all dimensions explicitly passed as tuple
    a = np.random.randn(3, 3).astype(np.float32)
    axis = (0, 1)
    dtype = np.dtype('float32')
    keepdims = False
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 9, 1D float64 array with specified elements excluded via 'where'
    a = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    axis = (0,)
    dtype = np.dtype('float64')
    keepdims = True
    where = np.array([True, False, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 10, 2D float32 array with custom mask configuration
    a = np.random.randn(3, 2).astype(np.float32)
    axis = (1,)
    dtype = np.dtype('float32')
    keepdims = True
    where = np.array([[True, True], [False, True], [True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.mean_2"] = jax_numpy_mean_inputs()

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
