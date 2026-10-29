
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reduction over axis (0,)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True, True, True], dtype=bool)
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

    # Input 2: 2D array, reduction along axis (1,), keepdims=True
    a = np.random.randn(4, 3).astype(np.float32)
    axis = (1,)
    b = np.ones((1, 3), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((4, 3), dtype=bool)
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

    # Input 3: 2D array with float64, return_sign=True, negative weights in b
    a = np.random.randn(3, 4).astype(np.float64)
    axis = (0,)
    b = np.array(
        [
            [-1.0, 2.0, -0.5, 1.5],
            [2.0, -1.0, 1.0, -2.0],
            [0.5, 0.5, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    keepdims = False
    return_sign = True
    where = np.ones((3, 4), dtype=bool)
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

    # Input 4: 3D array, reduction over multiple axes (0, 2), broadcasted b
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    b = np.random.uniform(0.1, 2.0, size=(2, 1, 4)).astype(np.float32)
    keepdims = True
    return_sign = True
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.8, 0.2])
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

    # Input 5: 1D array, axis=(-1,), scalar-like 1D b
    a = np.array([-10.0, -2.5, 0.0, 5.5, 12.0], dtype=np.float64)
    axis = (-1,)
    b = np.array([2.0], dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.array([True, True, False, True, True], dtype=bool)
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

    # Input 6: 4D array, reduction over middle axes (1, 2)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (1, 2)
    b = np.ones((1, 3, 4, 1), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3, 4, 5), dtype=bool)
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

    # Input 7: 2D array, reduce all axes (0, 1), keepdims=True
    a = np.random.randn(5, 5).astype(np.float32)
    axis = (0, 1)
    b = np.random.exponential(scale=1.0, size=(5, 5)).astype(np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((5, 5), dtype=bool)
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

    # Input 8: 3D array with all negative values, return_sign=True
    a = -np.random.uniform(1.0, 10.0, size=(3, 2, 4)).astype(np.float32)
    axis = (1,)
    b = np.random.uniform(0.5, 1.5, size=(3, 2, 4)).astype(np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((3, 2, 4), dtype=bool)
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

    # Input 9: 2D array, negative axis index (-2,), broadcasted where
    a = np.random.randn(6, 7).astype(np.float64)
    axis = (-2,)
    b = np.array([[1.5]], dtype=np.float64)
    keepdims = True
    return_sign = False
    where = np.ones((1, 7), dtype=bool)
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

    # Input 10: 5D array, reduction over axes (0, 2, 4), return_sign=True
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (0, 2, 4)
    b = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((2, 2, 2, 2, 2), dtype=bool)
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


generated_inputs["jax.scipy.special.logsumexp_2"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.logsumexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.logsumexp_2'.")


check_valid('jax.scipy.special.logsumexp', generated_inputs['jax.scipy.special.logsumexp_2'], lib="jax", suffix=2)
