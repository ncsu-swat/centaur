
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, standard case
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True, True, True])
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 2: 2D array, keepdims=True, return_sign=True, negative scaling factors b
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = [1]
    b = np.array([[-1.0, 2.0], [1.0, -3.0]], dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.array([[True, True], [True, True]])
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 3: 3D array, float64, reducing multiple axes, some where mask elements are False
    a = np.random.randn(2, 3, 2).astype(np.float64)
    axis = [0, 2]
    b = np.ones((2, 3, 2), dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.array([[[True, False], [True, True], [False, True]],
                      [[True, True], [False, True], [True, False]]])
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 4: 2D array, broadcasting b
    a = np.random.randn(4, 5).astype(np.float32)
    axis = [0]
    b = np.random.randn(1, 5).astype(np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((4, 5), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 5: 4D array, float32, keepdims=False, return_sign=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [1, 3]
    b = np.random.randn(2, 2, 2, 2).astype(np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 6: 1D array of float64, keepdims=True
    a = np.array([-10.0, -20.0, -30.0], dtype=np.float64)
    axis = [0]
    b = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    keepdims = True
    return_sign = False
    where = np.array([True, True, False])
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 7: 3D array, axis contains negative index, return_sign=True
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = [-1]
    b = np.random.randn(3, 3, 3).astype(np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((3, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 8: 2D array with large elements to test logsumexp stability
    a = np.array([[1000.0, 1000.0], [1000.0, 1000.0]], dtype=np.float32)
    axis = [0, 1]
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.ones((2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 9: 1D array, return_sign=True with completely negative sums
    a = np.array([0.0, 0.0], dtype=np.float32)
    axis = [0]
    b = np.array([-1.0, -2.0], dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.array([True, True])
    list_of_inputs.append({
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    })

    # Input 10: 3D array, broadcasting b on multiple dimensions
    a = np.random.randn(2, 4, 3).astype(np.float32)
    axis = [1]
    b = np.random.randn(1, 4, 1).astype(np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((2, 4, 3), dtype=bool)
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
