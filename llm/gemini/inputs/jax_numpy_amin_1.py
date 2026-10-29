
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def amin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([3.0, 1.0, 4.0, 1.5, 9.0], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
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

    # Input 2: 2D float32 array, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
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

    # Input 3: 2D int32 array with negative values
    a = np.array([[-5, 10, -2], [3, -8, 7]], dtype=np.int32)
    axis = 0
    keepdims = False
    initial = np.array(50, dtype=np.int32)
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

    # Input 4: 3D float64 array, negative axis
    a = np.random.randn(2, 3, 5).astype(np.float64)
    axis = -1
    keepdims = False
    initial = np.array(1e9, dtype=np.float64)
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

    # Input 5: 3D int64 array, keepdims=True
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 12
    axis = 1
    keepdims = True
    initial = np.array(100, dtype=np.int64)
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

    # Input 6: 4D float32 array
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 4)).astype(np.float32)
    axis = 2
    keepdims = False
    initial = np.array(20.0, dtype=np.float32)
    where = a < 5.0
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

    # Input 7: 1D int32 array with keepdims=True and negative axis
    a = np.array([42, -100, 15, 0, -200], dtype=np.int32)
    axis = -1
    keepdims = True
    initial = np.array(500, dtype=np.int32)
    where = np.array([True, False, True, True, False], dtype=bool)
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

    # Input 8: 2D int64 array
    a = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    axis = 0
    keepdims = True
    initial = np.array(100, dtype=np.int64)
    where = np.array([[True, True], [False, True], [True, False]], dtype=bool)
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

    # Input 9: 5D float32 array
    a = np.random.randn(2, 1, 3, 2, 4).astype(np.float32)
    axis = 4
    keepdims = False
    initial = np.array(np.inf, dtype=np.float32)
    where = np.ones((2, 1, 3, 2, 4), dtype=bool)
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

    # Input 10: 2D float64 array with keepdims=True
    a = np.array([[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5]], dtype=np.float64)
    axis = -2
    keepdims = True
    initial = np.array(0.0, dtype=np.float64)
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
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


generated_inputs["jax.numpy.amin_1"] = amin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amin_1'.")


check_valid('jax.numpy.amin', generated_inputs['jax.numpy.amin_1'], lib="jax", suffix=1)
