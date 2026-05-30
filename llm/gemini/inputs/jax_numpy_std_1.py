
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def std_inputs():
    list_of_inputs = []

    def make_input(a, axis, dtype, ddof, keepdims, where, mean, correction):
        return {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction
        }

    # Input 1: 2D array, axis=0, float32, ddof=0, keepdims=False
    a = np.random.randn(4, 5).astype(np.float32)
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 2: 2D array, axis=1, float64, ddof=1, keepdims=True
    a = np.random.randn(3, 6).astype(np.float64)
    axis = 1
    dtype = np.float64
    ddof = 1
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 3: 3D array, axis=2, float32, ddof=0, keepdims=True, with partial where
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.random.choice([True, True, False], size=a.shape)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 4: 2D array, negative axis, float32, ddof=1, keepdims=False
    a = np.random.randn(5, 5).astype(np.float32)
    axis = -1
    dtype = np.float32
    ddof = 1
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 5: 3D array, axis=0, float32, ddof=2, keepdims=True
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = 0
    dtype = np.float32
    ddof = 2
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 6: Float16 input, float32 output dtype, axis=1
    a = np.random.randn(4, 4).astype(np.float16)
    axis = 1
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True).astype(np.float16)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 7: Larger dimensions, axis=1, ddof=1, keepdims=False
    a = np.random.randn(10, 10, 10).astype(np.float32)
    axis = 1
    dtype = np.float32
    ddof = 1
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 8: 2D array, axis=0, float64, ddof=0, keepdims=True
    a = np.random.randn(8, 2).astype(np.float64)
    axis = 0
    dtype = np.float64
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 9: 2D array, axis=0, with partial where and ddof=1
    a = np.random.randn(6, 6).astype(np.float32)
    axis = 0
    dtype = np.float32
    ddof = 1
    keepdims = False
    where = np.random.choice([True, True, False], size=a.shape)
    mean = np.mean(a, axis=axis, keepdims=True, where=where)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    # Input 10: Simple 2D array, axis=1, ddof=0, keepdims=False, float64
    a = np.random.randn(5, 3).astype(np.float64)
    axis = 1
    dtype = np.float64
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = None
    list_of_inputs.append(make_input(a, axis, dtype, ddof, keepdims, where, mean, correction))

    return list_of_inputs

generated_inputs["jax.numpy.std_1"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_1'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_1'], lib="jax", suffix=1)
