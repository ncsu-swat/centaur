
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sum_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [1]
    dtype = np.dtype('float32')
    keepdims = False
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

    # Input 2
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    axis = [0, 2]
    dtype = np.dtype('int64')
    keepdims = True
    initial = np.array(5, dtype=np.int64)
    where = np.random.choice([True, False], size=(2, 3, 4))
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

    # Input 3
    a = np.random.randn(10).astype(np.float64)
    axis = [0]
    dtype = np.dtype('float64')
    keepdims = False
    initial = np.array(-1.0, dtype=np.float64)
    where = np.ones((10,), dtype=bool)
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

    # Input 4
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [1, 3]
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
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

    # Input 5
    a = np.random.randint(-5, 5, size=(5, 5)).astype(np.int16)
    axis = [0]
    dtype = np.dtype('int32')
    keepdims = False
    initial = np.array(10, dtype=np.int32)
    where = np.ones((5, 5), dtype=bool)
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

    # Input 6
    a = np.random.randn(4, 1, 3).astype(np.float32)
    axis = [2]
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(-0.5, dtype=np.float32)
    where = np.ones((4, 1, 3), dtype=bool)
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

    # Input 7
    a = np.random.randint(0, 5, size=(5,)).astype(np.int32)
    axis = [0]
    dtype = np.dtype('int32')
    keepdims = True
    initial = np.array(0, dtype=np.int32)
    where = np.array([True, False, True, False, True], dtype=bool)
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

    # Input 8
    a = np.random.randn(2, 3).astype(np.float64)
    axis = [0, 1]
    dtype = np.dtype('float64')
    keepdims = False
    initial = np.array(100.0, dtype=np.float64)
    where = np.ones((2, 3), dtype=bool)
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

    # Input 9
    a = np.random.randn(1, 1, 1).astype(np.float32)
    axis = [0, 1, 2]
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((1, 1, 1), dtype=bool)
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

    # Input 10
    a = np.random.randint(-20, 20, size=(3, 2)).astype(np.int64)
    axis = [1]
    dtype = np.dtype('int64')
    keepdims = False
    initial = np.array(-10, dtype=np.int64)
    where = np.ones((3, 2), dtype=bool)
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

generated_inputs["jax.numpy.sum_3"] = jax_numpy_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_3'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_3'], lib="jax", suffix=3)
