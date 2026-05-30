
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def min_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, axis [0], keepdims True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2: 3D array, int32, axis [1, 2], keepdims False
    a = np.random.randint(-100, 100, size=(2, 3, 2)).astype(np.int32)
    axis = [1, 2]
    keepdims = False
    initial = np.array(200, dtype=np.int32)
    where = np.ones((2, 3, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3: 1D array, float64, axis [0], keepdims True, negative values
    a = np.array([-5.5, 3.2, -10.1, 0.0, 4.4], dtype=np.float64)
    axis = [0]
    keepdims = True
    initial = np.array(1.0, dtype=np.float64)
    where = np.array([True, False, True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4: 2D array, int64, axis [0, 1], keepdims False
    a = np.random.randint(-10, 10, size=(4, 4)).astype(np.int64)
    axis = [0, 1]
    keepdims = False
    initial = np.array(100, dtype=np.int64)
    where = np.ones((4, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5: 2D array, float32, axis [1], keepdims True, some False in where
    a = np.random.randn(2, 4).astype(np.float32)
    axis = [1]
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.array([[True, True, False, False], [False, True, True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6: 2D array (1x1), float32, axis [0, 1], keepdims False
    a = np.array([[1.5]], dtype=np.float32)
    axis = [0, 1]
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7: 3D array, int32, axis [2], keepdims True
    a = np.random.randint(-50, 50, size=(3, 2, 4)).astype(np.int32)
    axis = [2]
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.ones((3, 2, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8: 4D array, float64, axis [0, 2], keepdims False
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    axis = [0, 2]
    keepdims = False
    initial = np.array(50.0, dtype=np.float64)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9: 1D array, int16, axis [0], keepdims True
    a = np.random.randint(-20, 20, size=(10,)).astype(np.int16)
    axis = [0]
    keepdims = True
    initial = np.array(100, dtype=np.int16)
    where = np.ones((10,), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10: 2D array, float32, axis [1], keepdims False, mixed where
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    axis = [1]
    keepdims = False
    initial = np.array(2.0, dtype=np.float32)
    where = np.array([[True, True, True], [True, False, True], [False, True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.min_3"] = min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.min_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.min_3'.")


check_valid('jax.numpy.min', generated_inputs['jax.numpy.min_3'], lib="jax", suffix=3)
