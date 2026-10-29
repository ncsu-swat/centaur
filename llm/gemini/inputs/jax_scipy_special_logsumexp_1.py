
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic reduction along axis 0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones(5, dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 2: 2D array, keepdims=True, reduction along axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    b = np.ones((3, 4), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 3: 2D array, axis=1, return_sign=True with signed scaling factors
    a = np.random.randn(4, 3).astype(np.float32)
    axis = 1
    b = np.array(
        [[-1.0, 2.0, -0.5], [1.0, -1.0, 2.0], [0.5, 0.5, -1.0], [2.0, -3.0, 1.0]],
        dtype=np.float32,
    )
    keepdims = False
    return_sign = True
    where = np.ones((4, 3), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 4: 3D array, axis=-1, broadcastable b
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = -1
    b = np.ones((1, 1, 4), dtype=np.float64)
    keepdims = True
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 5: 3D array, axis=1, broadcastable where mask
    a = np.random.randn(3, 5, 2).astype(np.float32)
    axis = 1
    b = np.full((3, 5, 2), 2.0, dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.array([[[True], [True], [False], [True], [True]]], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 6: 1D array with negative values in a, scalar-like b
    a = np.array([-10.0, -20.0, -30.0, -5.0], dtype=np.float32)
    axis = 0
    b = np.array([0.5], dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 7: 4D array, axis=2, float32
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    axis = 2
    b = np.ones((2, 2, 3, 2), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 2, 3, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 8: 2D array, large magnitude values to test numerical stability
    a = np.array([[1000.0, 1001.0], [1002.0, 1003.0]], dtype=np.float64)
    axis = 1
    b = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 9: 2D array, axis=-2 (negative axis index)
    a = np.random.randn(5, 4).astype(np.float32)
    axis = -2
    b = np.ones((5, 1), dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.ones((5, 4), dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    # Input 10: 1D array, axis=-1, float64
    a = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    axis = -1
    b = np.linspace(0.1, 1.0, 10, dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.array([i % 2 == 0 for i in range(10)], dtype=bool)
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "b": b,
            "keepdims": keepdims,
            "return_sign": return_sign,
            "where": where,
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.special.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.logsumexp_1'.")


check_valid('jax.scipy.special.logsumexp', generated_inputs['jax.scipy.special.logsumexp_1'], lib="jax", suffix=1)
