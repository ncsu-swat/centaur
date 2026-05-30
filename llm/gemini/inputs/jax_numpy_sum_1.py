
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sum_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 2D array, sum over axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int32 2D array, sum over axis 1
    a = np.random.randint(-10, 10, size=(5, 2)).astype(np.int32)
    axis = 1
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(10, dtype=np.int32)
    where = np.ones((5, 2), dtype=bool)
    promote_integers = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, sum over axis 2, with non-trivial mask
    a = np.random.randn(2, 3, 2).astype(np.float64)
    axis = 2
    dtype = np.dtype(np.float64)
    keepdims = True
    initial = np.array(-1.5, dtype=np.float64)
    where = np.array([[[True, False], [True, True], [False, True]], 
                      [[False, False], [True, True], [True, True]]], dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int16 array, sum over axis 0, promote integers
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16)
    axis = 0
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([True, False, True, False, True, False, True, False, True, False], dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, sum over axis 1
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(0.5, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, negative axis
    a = np.random.randn(4, 3).astype(np.float32)
    axis = -1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((4, 3), dtype=bool)
    promote_integers = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array, sum over axis 0 with custom dtype
    a = np.random.randint(-5, 5, size=(3, 3, 3)).astype(np.int32)
    axis = 0
    dtype = np.dtype(np.int64)
    keepdims = True
    initial = np.array(-5, dtype=np.int64)
    where = np.ones((3, 3, 3), dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, keepdims=False
    a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.2, dtype=np.float32)
    where = np.array([True, False, True, False, True], dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 column vector array, axis=0
    a = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.int32)
    axis = 0
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.ones((6, 1), dtype=bool)
    promote_integers = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array, negative axis sum
    a = np.random.randn(2, 4).astype(np.float64)
    axis = -2
    dtype = np.dtype(np.float64)
    keepdims = False
    initial = np.array(0.001, dtype=np.float64)
    where = np.ones((2, 4), dtype=bool)
    promote_integers = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims,
        "initial": initial,
        "where": where,
        "promote_integers": promote_integers
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sum_1"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_1'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_1'], lib="jax", suffix=1)
