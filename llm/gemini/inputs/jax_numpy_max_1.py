
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.5, -2.3, 4.8, 0.2], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.array([True, False, True, True], dtype=bool)
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

    # Input 2: 2D int32 array, keepdims=True
    a = np.array([[10, 20, 5], [-5, 12, 0]], dtype=np.int32)
    axis = 1
    keepdims = True
    initial = np.array(-999, dtype=np.int32)
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

    # Input 3: 2D float64 array, negative values, initial=-inf
    a = np.array([[-12.5, -3.2], [-5.0, -1.1]], dtype=np.float64)
    axis = 0
    keepdims = False
    initial = np.array(-np.inf, dtype=np.float64)
    where = np.array([[True, True], [False, True]], dtype=bool)
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

    # Input 4: 3D int64 array, negative axis
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 10
    axis = -1
    keepdims = True
    initial = np.array(-50, dtype=np.int64)
    where = np.ones((2, 3, 4), dtype=bool)
    where[0, 1, 2] = False
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

    # Input 5: 1D int16 array
    a = np.array([-32000, 1500, 32000, -100], dtype=np.int16)
    axis = -1
    keepdims = False
    initial = np.array(-32768, dtype=np.int16)
    where = np.array([True, True, False, True], dtype=bool)
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

    # Input 6: 4D float32 array, broadcasting where
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = 2
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((1, 1, 4, 1), dtype=bool)
    where[0, 0, 1, 0] = False
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

    # Input 7: 2D uint8 array
    a = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    axis = 0
    keepdims = True
    initial = np.array(0, dtype=np.uint8)
    where = np.array([[True, False], [True, True]], dtype=bool)
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

    # Input 8: 3D float16 array
    a = np.linspace(-5.0, 5.0, 12, dtype=np.float16).reshape((2, 2, 3))
    axis = 1
    keepdims = False
    initial = np.array(-100.0, dtype=np.float16)
    where = np.ones((2, 2, 3), dtype=bool)
    where[1, 0, :] = False
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

    # Input 9: 2D float32 array with all elements unmasked
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    axis = 1
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((3, 2), dtype=bool)
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

    # Input 10: 5D int32 array
    a = np.random.randint(-100, 100, size=(2, 2, 2, 2, 2), dtype=np.int32)
    axis = 3
    keepdims = True
    initial = np.array(-200, dtype=np.int32)
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

    return list_of_inputs


generated_inputs["jax.numpy.max_1"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_1'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_1'], lib="jax", suffix=1)
