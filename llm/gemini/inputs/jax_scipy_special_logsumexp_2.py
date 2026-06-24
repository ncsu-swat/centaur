
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32 input, full reduction
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True, True, True], dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, reduction over axis 1, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    b = np.abs(np.random.randn(3, 4)).astype(np.float32)
    keepdims = True
    return_sign = True
    where = np.random.choice([True, False], size=(3, 4))
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, reduction over multiple axes
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = (0, 2)
    b = np.ones((2, 3, 4), dtype=np.float64)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative scaling values for 'b' when return_sign is True
    a = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    axis = (0,)
    b = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.array([True, True, True], dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, reduction over axis 2
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (2,)
    b = np.ones((2, 2, 2, 2), dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with some 'where' elements excluded
    a = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    axis = (0,)
    b = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger dimensions, float32, full reduction
    a = np.random.randn(10, 10, 10).astype(np.float32)
    axis = (0, 1, 2)
    b = np.random.rand(10, 10, 10).astype(np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((10, 10, 10), dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 with keepdims=True
    a = np.random.randn(5, 5).astype(np.float32)
    axis = (1,)
    b = np.ones((5, 5), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((5, 5), dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 with large values and return_sign=True
    a = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    axis = (0,)
    b = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    keepdims = False
    return_sign = True
    where = np.array([True, True, True], dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array with all-False mask
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = (1, 2)
    b = np.ones((3, 3, 3), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.zeros((3, 3, 3), dtype=bool)
    input_dict = {
        "a": a, "axis": axis, "b": b, "keepdims": keepdims, 
        "return_sign": return_sign, "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
