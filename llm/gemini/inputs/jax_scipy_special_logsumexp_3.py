
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reduce all along axis [0]
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 2: 2D array, reduce along axis [1], keepdims True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [1]
    b = np.ones((3, 4), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 3: 2D array, reduce along axis [0], return_sign True with negative b
    a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    axis = [0]
    b = np.array([[-1.0, 1.0], [2.0, -1.0]], dtype=np.float64)
    keepdims = False
    return_sign = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 4: 3D array, reduce along multiple axes [0, 2]
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = [0, 2]
    b = np.ones((2, 3, 4), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 5: 3D array with masking via `where`
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 10.0
    axis = [1]
    b = np.full((2, 3, 4), 0.5, dtype=np.float32)
    keepdims = True
    return_sign = False
    where = (a > 0)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 6: Broadcastable `b` and `where`
    a = np.random.randn(4, 5).astype(np.float32)
    axis = [1]
    b = np.array([[1.0], [2.0], [1.0], [0.5]], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([[True], [True], [False], [True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 7: 4D array with negative axis indexing
    a = np.random.randn(2, 2, 3, 3).astype(np.float64)
    axis = [-1, -2]
    b = np.ones((2, 2, 3, 3), dtype=np.float64)
    keepdims = True
    return_sign = True
    where = np.ones((2, 2, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 8: Reduce empty axis list (no reduction)
    a = np.random.randn(3, 3).astype(np.float32)
    axis = []
    b = np.ones((3, 3), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 9: Large negative numbers with b containing zeros
    a = np.array([-1000.0, -2000.0, -3000.0], dtype=np.float32)
    axis = [0]
    b = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.array([False, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    # Input 10: 5D array reduction
    a = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    axis = [0, 2, 4]
    b = np.ones((2, 1, 3, 1, 2), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((2, 1, 3, 1, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims,
        "return_sign": return_sign, "where": where
    })

    return list_of_inputs

generated_inputs["jax.scipy.special.logsumexp_3"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.logsumexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.logsumexp_3'.")


check_valid('jax.scipy.special.logsumexp', generated_inputs['jax.scipy.special.logsumexp_3'], lib="jax", suffix=3)
