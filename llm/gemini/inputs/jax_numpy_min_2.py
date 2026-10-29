
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([2.5, -1.0, 3.2, -5.5, 0.0], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 2: 2D int32 array, reducing along axis 1 with keepdims=True
    a = np.array([[3, -2, 7], [1, 0, -8]], dtype=np.int32)
    axis = (1,)
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 3: 2D float64 array, multi-axis reduction (0, 1)
    a = np.array(
        [[-3.14, 2.71, 0.0], [1.41, -0.57, 9.99], [4.2, -8.1, 1.2]],
        dtype=np.float64,
    )
    axis = (0, 1)
    keepdims = False
    initial = np.array(50.0, dtype=np.float64)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 4: 3D int64 array with broadcasted where
    a = np.arange(-12, 12, dtype=np.int64).reshape((2, 3, 4))
    axis = (2,)
    keepdims = False
    initial = np.array(999, dtype=np.int64)
    where = np.array([[[True, False, True, False]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 5: 3D float32 array, negative axis index
    a = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (-1,)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = a > 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 6: 4D float32 array reducing two non-adjacent axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (0, 2)
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 3, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 7: 1D int16 array with keepdims=True
    a = np.array([-300, 150, -500, 200, 0], dtype=np.int16)
    axis = (-1,)
    keepdims = True
    initial = np.array(1000, dtype=np.int16)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 8: 2D uint8 array reducing along axis 0
    a = np.array([[10, 200, 50], [80, 5, 120]], dtype=np.uint8)
    axis = (0,)
    keepdims = False
    initial = np.array(255, dtype=np.uint8)
    where = np.array([[True, False, True], [True, True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 9: 3D float64 array reducing axes (1, 2) with keepdims=True
    a = np.linspace(-10.0, 10.0, 24, dtype=np.float64).reshape((2, 3, 4))
    axis = (1, 2)
    keepdims = True
    initial = np.array(100.0, dtype=np.float64)
    where = np.array([[[True], [False], [True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 10: 2D float32 array with 1D broadcastable where
    a = np.array(
        [[1.0, -2.0, 3.0, -4.0], [5.0, 6.0, -7.0, 8.0]], dtype=np.float32
    )
    axis = (0,)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

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
