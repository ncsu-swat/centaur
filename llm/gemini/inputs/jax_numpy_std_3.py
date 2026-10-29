
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def std_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, keepdims=False
    a = np.array([1.5, -2.3, 3.7, 4.1, -0.5, 6.2], dtype=np.float32)
    axis = [0]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float32)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, axis=[1], keepdims=True, correction=1
    a = np.random.randn(4, 8).astype(np.float64)
    axis = [1]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, int32 input with float32 dtype output, axis=[0]
    a = np.random.randint(-10, 10, size=(6, 5)).astype(np.int32)
    axis = [0]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float32)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float32, multiple axes reduction [0, 2], keepdims=True
    a = np.random.uniform(-5.0, 5.0, size=(4, 3, 6)).astype(np.float32)
    axis = [0, 2]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float32)
    where = np.ones_like(a, dtype=bool)
    where[0, :, 0] = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative axis index [-1], keepdims=False
    a = np.random.randn(3, 5, 7).astype(np.float64)
    axis = [-1]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=[1, 3], keepdims=False
    a = np.random.randn(2, 4, 3, 5).astype(np.float32)
    axis = [1, 3]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float32)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float64, axis=[0, 1, 2], keepdims=True
    a = np.random.randn(3, 3, 4, 2).astype(np.float64)
    axis = [0, 1, 2]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, masked where elements
    a = np.array(
        [[10.0, -2.0, 3.5, 4.0], [5.0, 6.0, -7.0, 8.5], [9.0, 1.2, 3.4, -5.6]],
        dtype=np.float32,
    )
    axis = [0]
    where = np.array(
        [[True, True, False, True], [True, False, True, True], [False, True, True, True]],
        dtype=bool,
    )
    # Broadcast-compatible mean
    mean = (np.sum(a * where, axis=tuple(axis), keepdims=True) / np.sum(where, axis=tuple(axis), keepdims=True)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float32),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, float64, axis=[0, 2, 4], keepdims=True
    a = np.random.randn(2, 3, 2, 4, 2).astype(np.float64)
    axis = [0, 2, 4]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, large magnitude values, axis=[1], keepdims=False
    a = np.random.uniform(1e3, 1e5, size=(5, 10)).astype(np.float64)
    axis = [1]
    mean = np.mean(a, axis=tuple(axis), keepdims=True).astype(np.float64)
    where = np.ones_like(a, dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": np.dtype(np.float64),
        "ddof": 0,
        "keepdims": False,
        "where": where,
        "mean": mean,
        "correction": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.std_3"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_3'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_3'], lib="jax", suffix=3)
