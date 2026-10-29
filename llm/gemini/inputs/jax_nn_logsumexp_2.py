
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, reduction along axis (0,)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = (0,)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 2: 2D array, float32, keepdims=True, with partial mask
    a = np.array([[-1.5, 2.3, 0.0], [4.1, -2.2, 1.1]], dtype=np.float32)
    axis = (1,)
    b = np.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]], dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.array([[True, True, False], [True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 3: 2D array, float64, return_sign=True with signed scaling factors
    a = np.array([[2.0, 3.0], [1.0, 5.0]], dtype=np.float64)
    axis = (0,)
    b = np.array([[-1.0, 1.0], [2.0, -1.5]], dtype=np.float64)
    keepdims = False
    return_sign = True
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 4: 3D array, axis multiple dimensions (1, 2)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (1, 2)
    b = np.ones((2, 3, 4), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 5: 3D array, negative axis (-1,), broadcastable b
    a = np.random.randn(4, 5, 6).astype(np.float32)
    axis = (-1,)
    b = np.ones((1, 1, 6), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((4, 5, 6), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 6: 4D array, axis=(0, 2), return_sign=True, float64
    a = np.random.randn(2, 3, 2, 4).astype(np.float64)
    axis = (0, 2)
    b = np.random.randn(2, 3, 2, 4).astype(np.float64)
    keepdims = True
    return_sign = True
    where = np.random.choice([True, False], size=(2, 3, 2, 4), p=[0.8, 0.2])
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 7: 1D array with negative values and float64
    a = np.array([-10.0, -20.0, -30.0, -40.0], dtype=np.float64)
    axis = (0,)
    b = np.array([2.0, 1.5, 0.5, 3.0], dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 8: 2D array, reduce all axes (0, 1)
    a = np.array([[0.5, -0.5, 1.2], [-2.0, 3.1, -1.0]], dtype=np.float32)
    axis = (0, 1)
    b = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 9: 3D array, axis=(0,), broadcastable where and b
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = (0,)
    b = np.ones((1, 4, 5), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((1, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 10: 2D array, axis=(-2,), return_sign=True, float32
    a = np.array([[10.0, -10.0], [5.0, -5.0], [0.0, 0.0]], dtype=np.float32)
    axis = (-2,)
    b = np.array([[-1.0, 2.0], [3.0, -0.5], [1.0, 1.0]], dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((3, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.nn.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logsumexp_2'.")


check_valid('jax.nn.logsumexp', generated_inputs['jax.nn.logsumexp_2'], lib="jax", suffix=2)
