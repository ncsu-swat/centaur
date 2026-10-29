
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    a = np.array([1.5, -2.3, 4.8, 0.0, -5.1], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-np.inf, dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
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

    # Input 2: 2D int32 array, reduce along axis 1 with keepdims=True
    a = np.array([[3, -1, 4], [1, 5, -9]], dtype=np.int32)
    axis = (1,)
    keepdims = True
    initial = np.array(-100, dtype=np.int32)
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

    # Input 3: 2D float64 array, reduce over all axes (0, 1)
    a = np.array([[-10.5, 2.3], [4.1, -0.7]], dtype=np.float64)
    axis = (0, 1)
    keepdims = False
    initial = np.array(-1e9, dtype=np.float64)
    where = np.ones((2, 2), dtype=bool)
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

    # Input 4: 3D int64 array, reduce along axis 0
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 12
    axis = (0,)
    keepdims = True
    initial = np.array(-1000, dtype=np.int64)
    where = np.ones((2, 3, 4), dtype=bool)
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

    # Input 5: 3D float32 array, reduce along axes (1, 2)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = (1, 2)
    keepdims = False
    initial = np.array(-np.inf, dtype=np.float32)
    where = a > 0
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

    # Input 6: 4D float32 array, negative axis indexing (-1,)
    a = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 4)).astype(np.float32)
    axis = (-1,)
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((2, 2, 3, 4), dtype=bool)
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

    # Input 7: 2D int32 array with broadcastable where shape (1, 4)
    a = np.array([[10, -20, 30, -40], [5, -15, 25, -35]], dtype=np.int32)
    axis = (0,)
    keepdims = False
    initial = np.array(-32768, dtype=np.int32)
    where = np.array([[True, False, True, True]], dtype=bool)
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

    # Input 8: 1D int32 array, negative axis
    a = np.array([10, 50, 20, 90, 40], dtype=np.int32)
    axis = (-1,)
    keepdims = True
    initial = np.array(0, dtype=np.int32)
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

    # Input 9: 3D float64 array, reduce axes (0, 2) with broadcastable where shape (2, 1, 1)
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = (0, 2)
    keepdims = True
    initial = np.array(-1e6, dtype=np.float64)
    where = np.array([[[True]], [[False]]], dtype=bool)
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

    # Input 10: 2D float32 array, negative axis (-2,)
    a = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    axis = (-2,)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.array([[True, True, False], [False, True, True]], dtype=bool)
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


generated_inputs["jax.numpy.max_2"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_2'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_2'], lib="jax", suffix=2)
