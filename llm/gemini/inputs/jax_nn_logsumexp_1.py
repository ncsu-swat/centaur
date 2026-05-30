
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, reduction along axis 1, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.uniform(0.1, 1.0, size=(3, 4)).astype(np.float32)
    where = np.random.choice([True, False], size=(3, 4), p=[0.8, 0.2]).astype(bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, return_sign=True
    a = np.random.randn(2, 3, 4).astype(np.float64)
    b = np.random.randn(2, 3, 4).astype(np.float64)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative coordinates of axis
    a = np.random.randn(5, 5).astype(np.float32)
    b = np.ones((5, 5), dtype=np.float32)
    where = np.ones((5, 5), dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted b and where
    a = np.random.randn(4, 3).astype(np.float32)
    b = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
    where = np.array([[True], [False], [True], [True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher-dimensional array 4D
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    b = np.ones((2, 2, 3, 3), dtype=np.float32)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 3,
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 type and negative scaling factors for sign return
    a = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    where = np.array([[True, True], [True, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "b": b,
        "keepdims": False,
        "return_sign": True,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger scale dimensions
    a = np.random.randn(10, 20).astype(np.float32)
    b = np.random.uniform(0.1, 2.0, size=(10, 20)).astype(np.float32)
    where = np.random.choice([True, False], size=(10, 20), p=[0.9, 0.1]).astype(bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element dimensions broadcasted
    a = np.random.randn(1, 5, 1).astype(np.float32)
    b = np.ones((1, 5, 1), dtype=np.float32)
    where = np.ones((1, 5, 1), dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "b": b,
        "keepdims": True,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 with custom boolean mask
    a = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    where = np.array([False, True, False], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "b": b,
        "keepdims": False,
        "return_sign": False,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logsumexp_1'.")


check_valid('jax.nn.logsumexp', generated_inputs['jax.nn.logsumexp_1'], lib="jax", suffix=1)
