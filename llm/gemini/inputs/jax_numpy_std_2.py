
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def std_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(10).astype(np.float32)
    axis = (0,)
    dtype = None
    ddof = 0
    keepdims = False
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.randn(4, 4).astype(np.float64)
    axis = (0, 1)
    dtype = None
    ddof = 1
    keepdims = True
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(3, 5).astype(np.float32)
    axis = (1,)
    dtype = np.float32
    ddof = 1
    keepdims = False
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(4, 5).astype(np.float32)
    axis = (0,)
    dtype = None
    ddof = 0
    keepdims = False
    where = np.random.choice([True, False], size=(4, 5))
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = (1, 2)
    dtype = None
    ddof = 0
    keepdims = True
    where = None
    mean = np.mean(a, axis=(1, 2), keepdims=True)
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.random.randn(8).astype(np.float32)
    axis = (0,)
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.randn(5, 5).astype(np.float32)
    axis = (0,)
    dtype = np.float64
    ddof = 0
    keepdims = False
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.random.randn(6, 6).astype(np.float32)
    axis = (1,)
    dtype = None
    ddof = 0
    keepdims = True
    where = np.ones((6, 6), dtype=bool)
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    dtype = None
    ddof = 2
    keepdims = False
    where = None
    mean = None
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    dtype = np.float64
    ddof = 0
    keepdims = True
    where = None
    mean = np.mean(a, axis=(1,), keepdims=True)
    correction = None
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "ddof": ddof,
        "keepdims": keepdims,
        "where": where,
        "mean": mean,
        "correction": correction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.std_2"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_2'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_2'], lib="jax", suffix=2)
