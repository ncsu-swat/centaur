
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def amax_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(4, 5).astype(np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((4, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2
    a = np.random.randn(2, 3, 4).astype(np.float32) - 5.0
    axis = (1, 2)
    keepdims = True
    initial = np.array(-100.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3
    a = np.random.randint(-50, 50, size=(10,)).astype(np.int32)
    axis = (0,)
    keepdims = False
    initial = np.array(-999, dtype=np.int32)
    where = np.array([True, False, True, True, False, True, False, True, True, False])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4
    a = np.random.randn(2, 2, 3, 3).astype(np.float64)
    axis = (0, 2)
    keepdims = True
    initial = np.array(-1e5, dtype=np.float64)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    axis = (1,)
    keepdims = False
    initial = np.array(-50, dtype=np.int64)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6
    a = np.random.randn(3, 1, 4).astype(np.float32)
    axis = (0, 2)
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.ones((3, 1, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7
    a = np.random.randn(5).astype(np.float32)
    axis = (0,)
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([True, True, False, True, False])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8
    a = np.random.randn(2, 3).astype(np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(-5.5, dtype=np.float32)
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9
    a = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    axis = (2,)
    keepdims = True
    initial = np.array(-100, dtype=np.int32)
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (1, 3, 4)
    keepdims = False
    initial = np.array(-99.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.amax_2"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amax_2'.")


check_valid('jax.numpy.amax', generated_inputs['jax.numpy.amax_2'], lib="jax", suffix=2)
