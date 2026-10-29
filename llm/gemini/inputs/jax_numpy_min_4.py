
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduction along axis 0
    a = np.array([-2.5, 3.1, -0.8, 10.2, -5.0], dtype=np.float32)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    initial = np.array(100.0, dtype=np.float32)
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

    # Input 2: 2D float32 array, reduction along axis 1 with keepdims=True
    a = np.array(
        [[2.0, -1.0, 5.0, 3.0], [0.5, 1.2, -4.0, 8.0], [-3.0, 6.0, 2.0, -1.5]],
        dtype=np.float32,
    )
    axis = np.array([1], dtype=np.int32)
    keepdims = True
    initial = np.array(50.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
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

    # Input 3: 2D int32 array, reduction along axis 0
    a = np.array([[10, -5, 3], [4, 8, -12], [-1, 0, 7], [15, -2, 6]], dtype=np.int32)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    initial = np.array(999, dtype=np.int32)
    where = np.array(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
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

    # Input 4: 3D float64 array, multiple axes reduction
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4)) - 10.0
    axis = np.array([0, 2], dtype=np.int32)
    keepdims = True
    initial = np.array(1e9, dtype=np.float64)
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

    # Input 5: 3D int64 array with negative axis and broadcasted where condition
    a = np.random.randint(-100, 100, size=(3, 2, 5), dtype=np.int64)
    axis = np.array([-1], dtype=np.int32)
    keepdims = False
    initial = np.array(10000, dtype=np.int64)
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

    # Input 6: 4D float32 array, reduction along two internal axes
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = np.array([1, 2], dtype=np.int32)
    keepdims = False
    initial = np.array(1e6, dtype=np.float32)
    where = np.ones((2, 2, 3, 3), dtype=bool)
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

    # Input 7: 2D float32 array with diagonal where condition
    a = np.random.uniform(-5.0, 5.0, size=(5, 5)).astype(np.float32)
    axis = np.array([0], dtype=np.int32)
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.eye(5, dtype=bool)
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

    # Input 8: 1D int32 array
    a = np.array([12, -45, 2, 89, 3, 100, -4, 1], dtype=np.int32)
    axis = np.array([0], dtype=np.int32)
    keepdims = False
    initial = np.array(1000, dtype=np.int32)
    where = np.array(
        [True, False, True, False, True, False, True, False], dtype=bool
    )
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

    # Input 9: 3D float32 array, full reduction across all axes
    a = np.random.randn(4, 4, 4).astype(np.float32)
    axis = np.array([0, 1, 2], dtype=np.int32)
    keepdims = True
    initial = np.array(500.0, dtype=np.float32)
    where = np.ones((4, 4, 4), dtype=bool)
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

    # Input 10: 2D float64 array, negative axis reduction with column-broadcast where
    a = np.random.randn(6, 2).astype(np.float64)
    axis = np.array([-2], dtype=np.int32)
    keepdims = False
    initial = np.array(1e9, dtype=np.float64)
    where = np.array(
        [[True], [False], [True], [True], [False], [True]], dtype=bool
    )
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

    # Input 11: 5D float32 array with dimensions of size 1
    a = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    axis = np.array([1, 3], dtype=np.int32)
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.ones((1, 2, 1, 3, 2), dtype=bool)
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
