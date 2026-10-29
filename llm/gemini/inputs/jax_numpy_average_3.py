
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def average_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis list, matching weights shape, returned=False, keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = [0]
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
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

    # Input 2: 2D array, axis=[0], weights shape matching a.shape, returned=True, keepdims=True
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    axis = [0]
    weights = np.ones((2, 3), dtype=np.float32)
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

    # Input 3: 2D array, axis=[1], 1D weights along axis 1, returned=False, keepdims=True
    a = np.random.randn(3, 4).astype(np.float64)
    axis = [1]
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
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

    # Input 4: 2D array with negative values, axis=[0, 1], full weights shape, returned=True, keepdims=False
    a = np.array([[-2.5, 3.5], [1.0, -4.0]], dtype=np.float32)
    axis = [0, 1]
    weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
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

    # Input 5: 3D array, axis=[-1], weights shape matching a.shape, returned=False, keepdims=False
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    axis = [-1]
    weights = np.ones((2, 3, 4), dtype=np.float32)
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

    # Input 6: 3D array, axis=[0, 2], weights matching shape of reduced axes (2, 4), returned=True, keepdims=True
    a = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    axis = [0, 2]
    weights = np.ones((2, 4), dtype=np.float32)
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

    # Input 7: 1D integer array, axis=[0], weights integer, returned=True, keepdims=False
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    axis = [0]
    weights = np.array([1, 2, 3, 4], dtype=np.int32)
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

    # Input 8: 4D array, axis=[1, 3], weights matching full shape, returned=False, keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [1, 3]
    weights = np.ones((2, 2, 2, 2), dtype=np.float32)
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

    # Input 9: 2D array, negative axis=[-2], weights shape matching a.shape, returned=True, keepdims=False
    a = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64)
    axis = [-2]
    weights = np.array([[1.0, 2.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64)
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

    # Input 10: 3D array, axis=[1], weights 1D matching dimension along axis 1, returned=False, keepdims=False
    a = np.ones((3, 5, 2), dtype=np.float32)
    axis = [1]
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
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

    return list_of_inputs


generated_inputs["jax.numpy.average_3"] = average_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.average_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.average_3'.")


check_valid('jax.numpy.average', generated_inputs['jax.numpy.average_3'], lib="jax", suffix=3)
