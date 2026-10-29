
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def var_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, float32, correction=0
    a = np.array([1.0, -2.0, 3.5, 4.2, -1.1], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, float64, correction=1
    a = np.array(
        [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0], [-1.0, -2.0, -3.0]],
        dtype=np.float64,
    )
    axis = 1
    dtype = np.dtype(np.float64)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, float32, correction=0
    a = np.array([[1.5, -2.5, 3.0], [4.0, 5.5, -1.0]], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=2, float64, correction=0
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4)) - 10.0
    axis = 2
    dtype = np.dtype(np.float64)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=-1, float32, correction=1
    a = np.linspace(-5.0, 5.0, 18, dtype=np.float32).reshape((3, 2, 3))
    axis = -1
    dtype = np.dtype(np.float32)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=1, float32, correction=0
    a = np.ones((2, 4, 3, 2), dtype=np.float32) * 3.14
    axis = 1
    dtype = np.dtype(np.float32)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, axis=-2, float64, correction=0
    a = np.array(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64
    )
    axis = -2
    dtype = np.dtype(np.float64)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, axis=-1, float32, correction=0
    a = np.array([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    axis = -1
    dtype = np.dtype(np.float32)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, axis=0, float64, correction=1
    a = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float64,
    )
    axis = 0
    dtype = np.dtype(np.float64)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, axis=3, float64, correction=0
    a = np.arange(36, dtype=np.float64).reshape((2, 3, 2, 3))
    axis = 3
    dtype = np.dtype(np.float64)
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(dtype)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.var_1"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_1'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_1'], lib="jax", suffix=1)
