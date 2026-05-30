
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def min_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, keepdims=False
    a = np.random.randn(5).astype(np.float32)
    axis = np.array(0, dtype=np.int32)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2: 2D float32 array, reducing over axis 1, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = np.array(1, dtype=np.int32)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3: 3D int32 array with negative values
    a = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    axis = np.array(2, dtype=np.int32)
    keepdims = False
    initial = np.array(999, dtype=np.int32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4: 3D array reducing over multiple axes
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = np.array([0, 2], dtype=np.int32)
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5: 2D float64 array, partial where mask
    a = np.random.randn(6, 6).astype(np.float64)
    axis = np.array(0, dtype=np.int32)
    keepdims = False
    initial = np.array(50.0, dtype=np.float64)
    where = np.random.choice([True, False], size=(6, 6))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6: 1D int64 array with specified elements excluded
    a = np.array([1, -2, 3, -4, 5], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    keepdims = True
    initial = np.array(10, dtype=np.int64)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7: 4D float32 array
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = np.array(3, dtype=np.int32)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8: 2D array with a broadcast compatible where mask (3x1 mask for 3x4 input)
    a = np.random.randn(3, 4).astype(np.float32)
    axis = np.array(0, dtype=np.int32)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True], [False], [True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9: High-dimensional 5D array with singletons
    a = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    axis = np.array(2, dtype=np.int32)
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 1, 3, 1, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10: 2D array with large values
    a = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
    axis = np.array(1, dtype=np.int32)
    keepdims = False
    initial = np.array(10000.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.min_4"] = min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.min_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.min_4'.")


check_valid('jax.numpy.min', generated_inputs['jax.numpy.min_4'], lib="jax", suffix=4)
