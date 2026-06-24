
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 2: 2D array, negative values, return_sign=True
    a = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    b = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 3: float64 array, 3D
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 2
    b = np.ones((2, 3, 4), dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 4: where mask filtering some elements
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    axis = 0
    b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 5: Broadcastable scaling factor b
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 4)
    keepdims = True
    return_sign = True
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 6: Large values testing numerical stability
    a = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    axis = 0
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 7: Very small negative values
    a = np.array([-1000.0, -1001.0, -1002.0], dtype=np.float32)
    axis = 0
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 8: negative weights in b and return_sign=True
    a = np.array([1.0, 2.0], dtype=np.float32)
    axis = 0
    b = np.array([-1.0, 2.0], dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.array([True, True], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 9: 4D array with random values
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 3
    b = np.ones((2, 2, 2, 2), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 10: float16 array
    a = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float16)
    axis = 0
    b = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float16)
    keepdims = True
    return_sign = False
    where = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

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
