
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.ones_like(a)
    where = np.array([True, True, True])
    input_dict = {
        "a": a,
        "axis": (0,),
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, keepdims=True, reduction over axis 1
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.abs(np.random.randn(2, 3)).astype(np.float32)
    where = np.ones((2, 3), dtype=bool)
    where[0, 1] = False
    input_dict = {
        "a": a,
        "axis": (1,),
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, return_sign=True, reduction over axis (0, 2)
    a = np.random.randn(2, 3, 4).astype(np.float64)
    b = np.random.randn(2, 3, 4).astype(np.float64)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, negative a, reduction axis 0, negative b, return_sign=True
    a = -np.abs(np.random.randn(4, 4).astype(np.float32))
    b = -np.abs(np.random.randn(4, 4).astype(np.float32))
    where = np.ones((4, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0,),
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32, reduction over axis (1, 3), keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = np.ones_like(a)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64, return_sign=True, keepdims=True
    a = np.random.randn(10).astype(np.float64)
    b = np.random.randn(10).astype(np.float64)
    where = (a > 0)
    input_dict = {
        "a": a,
        "axis": (0,),
        "b": b,
        "keepdims": True,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32, entire reduction (0, 1)
    a = np.random.randn(5, 5).astype(np.float32)
    b = np.ones_like(a)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32, reduction over axis (2,), keepdims=True
    a = np.random.randn(3, 3, 3).astype(np.float32)
    b = np.ones_like(a)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": (2,),
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64, return_sign=True
    a = np.random.randn(3, 4).astype(np.float64)
    b = np.random.randn(3, 4).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": (1,),
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    b = np.ones_like(a)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": (2, 4),
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
