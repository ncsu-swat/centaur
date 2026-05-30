
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def min_inputs():
    list_of_inputs = []

    # Input 1: float32, 2D, axis (0,), keepdims=True
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2: int32, 1D, axis (0,), keepdims=False
    a = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    axis = (0,)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3: float64, 3D, axis (1, 2), keepdims=True
    a = np.random.randn(2, 3, 2).astype(np.float64)
    axis = (1, 2)
    keepdims = True
    initial = np.array(10.0, dtype=np.float64)
    where = np.ones((2, 3, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4: int64, 2D, axis (1,), keepdims=False
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    axis = (1,)
    keepdims = False
    initial = np.array(100, dtype=np.int64)
    where = np.array([[True, True, False], [False, True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5: float32, 3D, axis (0, 2), keepdims=True
    a = np.random.uniform(-10, 10, (3, 3, 3)).astype(np.float32)
    axis = (0, 2)
    keepdims = True
    initial = np.array(20.0, dtype=np.float32)
    where = np.ones((3, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6: float32, 2D, axis (1,), keepdims=True, with negative values
    a = np.array([[-1.5, 2.5], [0.5, -3.5]], dtype=np.float32)
    axis = (1,)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7: int32, 2D, axis (0, 1), keepdims=False
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    axis = (0, 1)
    keepdims = False
    initial = np.array(20, dtype=np.int32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8: float32, 2D, axis (-1,), keepdims=False
    a = np.random.randn(5, 5).astype(np.float32)
    axis = (-1,)
    keepdims = False
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9: int32, 1D, axis (0,), keepdims=True
    a = np.array([100], dtype=np.int32)
    axis = (0,)
    keepdims = True
    initial = np.array(200, dtype=np.int32)
    where = np.array([True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10: float64, 4D, axis (1, 3), keepdims=False
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    axis = (1, 3)
    keepdims = False
    initial = np.array(10.0, dtype=np.float64)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.min_2"] = min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.min_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.min_2'.")


check_valid('jax.numpy.min', generated_inputs['jax.numpy.min_2'], lib="jax", suffix=2)
