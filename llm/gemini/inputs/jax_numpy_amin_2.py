
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def amin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, single-axis reduction
    a = np.array([2.5, -1.2, 3.8, -4.5], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2: 2D int32 array, keepdims=True
    a = np.random.randint(-50, 50, size=(4, 5)).astype(np.int32)
    axis = (1,)
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.ones((4, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3: 2D float64 array, reduce over all axes via tuple
    a = np.random.randn(3, 3).astype(np.float64)
    axis = (0, 1)
    keepdims = False
    initial = np.array(0.0, dtype=np.float64)
    where = np.array([[True, False, True], [False, True, True], [True, True, False]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4: 3D float32 array, reduction over multiple axes with keepdims
    a = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    axis = (0, 2)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5: 3D int64 array, negative axis index
    a = np.random.randint(0, 100, size=(2, 4, 3)).astype(np.int64)
    axis = (-1,)
    keepdims = False
    initial = np.array(50, dtype=np.int64)
    where = np.random.choice([True, False], size=(2, 4, 3))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6: 4D float32 array, multiple non-adjacent axes
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = (1, 3)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 3, 2, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7: 2D uint8 array
    a = np.random.randint(0, 256, size=(5, 5)).astype(np.uint8)
    axis = (0,)
    keepdims = True
    initial = np.array(255, dtype=np.uint8)
    where = (a > 10)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8: 1D int16 array with broadcasting where mask
    a = np.array([-15, -2, 0, 7, -30], dtype=np.int16)
    axis = (0,)
    keepdims = True
    initial = np.array(0, dtype=np.int16)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9: 2D float32 array, negative axis tuple
    a = np.random.randn(4, 6).astype(np.float32)
    axis = (-2, -1)
    keepdims = True
    initial = np.array(2.0, dtype=np.float32)
    where = np.ones((4, 6), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10: 3D float64 array, broadcastable where mask
    a = np.random.randn(3, 4, 5).astype(np.float64)
    axis = (1,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float64)
    where = np.ones((1, 4, 1), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.amin_2"] = amin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amin_2'.")


check_valid('jax.numpy.amin', generated_inputs['jax.numpy.amin_2'], lib="jax", suffix=2)
