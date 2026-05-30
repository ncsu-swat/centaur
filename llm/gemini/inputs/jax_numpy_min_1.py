
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_min_inputs():
    list_of_inputs = []

    # Input 1, valid - 2D integer array, axis 1, keepdims True
    a = np.array([[2, 5, 1, 6],
                  [3, -7, -2, 4],
                  [8, -4, 1, -3]], dtype=np.int32)
    axis = 1
    keepdims = True
    initial = np.array(0, dtype=np.int32)
    where = np.array([[True, False, True, False],
                      [False, False, True, True],
                      [True, True, True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2, valid - 2D float32 array, axis 0, keepdims False
    a = np.random.randn(5, 5).astype(np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3, valid - 3D int64 array, axis 2, keepdims True
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int64)
    axis = 2
    keepdims = True
    initial = np.array(100, dtype=np.int64)
    where = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4, valid - 3D float64 array, axis 1, keepdims False
    a = np.random.randn(4, 3, 2).astype(np.float64)
    axis = 1
    keepdims = False
    initial = np.array(5.0, dtype=np.float64)
    where = np.random.choice([True, False], size=(4, 3, 2))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5, valid - 1D float32 array, axis 0, keepdims True
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    axis = 0
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, False], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6, valid - 3D int32 array, axis -1, keepdims False
    a = np.random.randint(-5, 5, size=(4, 4, 4)).astype(np.int32)
    axis = -1
    keepdims = False
    initial = np.array(10, dtype=np.int32)
    where = np.ones((4, 4, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7, valid - 4D float32 array, axis 3, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 3
    keepdims = True
    initial = np.array(2.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8, valid - 2D int32 array, axis 0, keepdims False
    a = np.array([[10, 20], [30, 40]], dtype=np.int32)
    axis = 0
    keepdims = False
    initial = np.array(50, dtype=np.int32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9, valid - 3D float32 array with singleton dimension, axis 2, keepdims True
    a = np.random.randn(3, 1, 4).astype(np.float32)
    axis = 2
    keepdims = True
    initial = np.array(1.5, dtype=np.float32)
    where = np.ones((3, 1, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10, valid - 1D int32 array, axis 0, keepdims False
    a = np.random.randint(0, 100, size=(10,)).astype(np.int32)
    axis = 0
    keepdims = False
    initial = np.array(200, dtype=np.int32)
    where = np.array([True] * 5 + [False] * 5, dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.min_1"] = jax_numpy_min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.min_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.min_1'.")


check_valid('jax.numpy.min', generated_inputs['jax.numpy.min_1'], lib="jax", suffix=1)
