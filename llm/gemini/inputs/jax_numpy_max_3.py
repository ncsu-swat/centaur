
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([3.0, -1.0, 4.0, -5.0, 9.0], dtype=np.float32)
    axis = [0]
    keepdims = False
    initial = np.array(-1e9, dtype=np.float32)
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

    # Input 2: 2D int32 array, negative values, keepdims=True
    a = np.array([[-10, -20, -5], [-3, -40, -15]], dtype=np.int32)
    axis = [1]
    keepdims = True
    initial = np.array(-100, dtype=np.int32)
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

    # Input 3: 3D float64 array, multi-axis reduction
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [0, 2]
    keepdims = False
    initial = np.array(-np.inf, dtype=np.float64)
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

    # Input 4: 2D float32 array with broadcastable where mask
    a = np.arange(12, dtype=np.float32).reshape(4, 3) - 6.0
    axis = [0]
    keepdims = False
    initial = np.array(-50.0, dtype=np.float32)
    where = np.array([[True], [False], [True], [True]], dtype=bool)
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

    # Input 5: 4D int64 array, multiple axes with keepdims
    a = np.random.randint(-100, 100, size=(2, 3, 2, 4)).astype(np.int64)
    axis = [1, 3]
    keepdims = True
    initial = np.array(-1000, dtype=np.int64)
    where = np.ones((2, 3, 2, 4), dtype=bool)
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

    # Input 6: 1D int16 array with negative index axis
    a = np.array([-300, 250, -1000, 500, 0], dtype=np.int16)
    axis = [-1]
    keepdims = True
    initial = np.array(-32768, dtype=np.int16)
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

    # Input 7: 3D float32 array, negative axis reduction
    a = np.linspace(-5.0, 5.0, 24, dtype=np.float32).reshape(2, 3, 4)
    axis = [-2]
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
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

    # Input 8: 2D uint8 array reducing all axes
    a = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    axis = [0, 1]
    keepdims = False
    initial = np.array(0, dtype=np.uint8)
    where = np.array(
        [[True, False, True], [False, True, False], [True, False, True]],
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

    # Input 9: 5D float32 array
    a = np.random.uniform(-1.0, 1.0, size=(1, 2, 3, 2, 2)).astype(np.float32)
    axis = [2, 4]
    keepdims = True
    initial = np.array(-2.0, dtype=np.float32)
    where = np.ones((1, 2, 3, 2, 2), dtype=bool)
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

    # Input 10: 2D float64 array with broadcastable where across columns
    a = np.array([[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5]], dtype=np.float64)
    axis = [0]
    keepdims = True
    initial = np.array(-100.0, dtype=np.float64)
    where = np.array([[True, False, True]], dtype=bool)
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


generated_inputs["jax.numpy.max_3"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_3'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_3'], lib="jax", suffix=3)
