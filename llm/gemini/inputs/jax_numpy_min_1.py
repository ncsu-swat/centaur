
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    a = np.array([2.5, -1.2, 3.8, -4.5, 0.0], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 2: 2D int32 array, axis 1, keepdims=True
    a = np.array([[2, 5, 1, 6], [3, -7, -2, 4], [8, -4, 1, -3]], dtype=np.int32)
    axis = 1
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 3: 2D float64 array, negative axis -1
    a = np.array([[-10.5, 20.2], [30.1, -40.9], [0.0, 5.5]], dtype=np.float64)
    axis = -1
    keepdims = False
    initial = np.array(50.0, dtype=np.float64)
    where = np.array([[True, False], [True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 4: 3D int64 array, axis 0, keepdims=False
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 10
    axis = 0
    keepdims = False
    initial = np.array(1000, dtype=np.int64)
    where = a > -5
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 5: 3D float32 array, axis 2, broadcasting where
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = 2
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((3, 4, 1), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 6: 4D float64 array, axis 1
    a = np.random.uniform(-10.0, 10.0, size=(2, 3, 2, 4)).astype(np.float64)
    axis = 1
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = a < 5.0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 7: 1D uint8 array, axis -1
    a = np.array([255, 128, 42, 0, 17], dtype=np.uint8)
    axis = -1
    keepdims = True
    initial = np.array(255, dtype=np.uint8)
    where = np.array([True, False, True, False, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 8: 2D float32 array with all False in where
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(99.0, dtype=np.float32)
    where = np.zeros((2, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 9: 3D int16 array, axis -2
    a = np.array(
        [[[5, -2], [3, 9]], [[-11, 4], [0, -1]], [[7, 8], [-6, 3]]],
        dtype=np.int16,
    )
    axis = -2
    keepdims = True
    initial = np.array(32000, dtype=np.int16)
    where = np.array([[[True, True], [False, True]]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

    # Input 10: 2D float32 array, negative axis -2
    a = np.linspace(-5.0, 5.0, 12, dtype=np.float32).reshape(4, 3)
    axis = -2
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True], [False], [True], [True]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "keepdims": keepdims,
            "initial": initial,
            "where": where,
        }
    )

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
