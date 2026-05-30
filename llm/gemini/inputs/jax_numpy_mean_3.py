
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, reduction along axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.random.choice([True, False], size=(3, 4), p=[0.8, 0.2])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 2: 3D array, keepdims=True, float64 dtype
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [1, 2]
    dtype = np.dtype(np.float64)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 3: 1D integer array reduced to float32
    a = np.random.randint(-100, 100, size=(10,)).astype(np.int32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.random.choice([True, False], size=(10,), p=[0.7, 0.3])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 4: 4D array with negative values and float32 dtype
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = [2, 3]
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 2, 3, 3), p=[0.85, 0.15])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 5: Broadcasted "where" mask, 2D input
    a = np.random.randint(-50, 50, size=(5, 5)).astype(np.int16)
    axis = [1]
    dtype = np.dtype(np.float64)
    keepdims = False
    where = np.random.choice([True, False], size=(5, 1), p=[0.8, 0.2])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 6: 3D array, reduction over disjoint axes
    a = np.random.randn(3, 1, 4).astype(np.float32)
    axis = [0, 2]
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.random.choice([True, False], size=(3, 1, 4), p=[0.95, 0.05])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 7: Float16 precision reduced to float32
    a = np.random.randn(5).astype(np.float16)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.ones((5,), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 8: Scale-adjusted floats with broadcasted row mask
    a = np.random.randn(4, 4).astype(np.float64) * 10.0
    axis = [0]
    dtype = np.dtype(np.float64)
    keepdims = False
    where = np.random.choice([True, False], size=(1, 4), p=[0.75, 0.25])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 9: Fully reduced 3D array with all-True mask
    a = np.random.randn(2, 2, 2).astype(np.float32)
    axis = [0, 1, 2]
    dtype = np.dtype(np.float32)
    keepdims = False
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    # Input 10: Larger 10x10 matrix reduction with high density mask
    a = np.random.randn(10, 10).astype(np.float32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = True
    where = np.random.choice([True, False], size=(10, 10), p=[0.9, 0.1])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.mean_3"] = jax_numpy_mean_inputs()

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
