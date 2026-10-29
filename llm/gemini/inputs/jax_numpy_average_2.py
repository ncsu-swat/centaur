
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def average_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis as 1-tuple, matching weights shape
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    weights = np.array([0.5, 1.0, 2.0, 1.5], dtype=np.float32)
    returned = False
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 2: 2D array, weights matching full shape, returned=True, keepdims=True
    a = np.array([[-2.0, 4.0], [3.0, -1.0]], dtype=np.float64)
    axis = (0,)
    weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    returned = True
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 3: 2D array, axis=(1,), 1D weights matching axis dimension
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    axis = (1,)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    returned = False
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 4: 3D array, axis=(0, 2), weights matching sub-shape
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    weights = np.ones((2, 4), dtype=np.float32)
    returned = True
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 5: 3D array, negative axis index
    a = np.random.uniform(1.0, 10.0, size=(4, 5, 6)).astype(np.float64)
    axis = (-1,)
    weights = np.random.uniform(0.1, 1.0, size=(6,)).astype(np.float64)
    returned = False
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 6: Integer inputs with full-shape weights
    a = np.array([[10, -20, 30], [-40, 50, -60]], dtype=np.int32)
    axis = (1,)
    weights = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    returned = True
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 7: 4D array, axis=(1, 2), matching sub-shape weights
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (1, 2)
    weights = np.random.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32)
    returned = False
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 8: 2D array, all axes reduced
    a = np.array([[1.5, -2.5], [3.5, 4.5]], dtype=np.float32)
    axis = (0, 1)
    weights = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    returned = True
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 9: Complex numbers input
    a = (
        np.array([[1 + 2j, 3 - 1j], [2 - 3j, -1 + 4j]], dtype=np.complex64)
    )
    axis = (0,)
    weights = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    returned = False
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    # Input 10: 3D array with negative axis and keepdims=True
    a = np.random.randn(3, 4, 5).astype(np.float64)
    axis = (-2, -1)
    weights = np.random.uniform(0.1, 1.0, size=(4, 5)).astype(np.float64)
    returned = True
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "weights": weights,
                "returned": returned,
                "keepdims": keepdims,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.average_2"] = average_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.average_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.average_2'.")


check_valid('jax.numpy.average', generated_inputs['jax.numpy.average_2'], lib="jax", suffix=2)
