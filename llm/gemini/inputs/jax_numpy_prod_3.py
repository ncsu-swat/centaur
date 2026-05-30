
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def prod_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, axis [0], keepdims False
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 2: 3D int32 array, axis [1], keepdims True
    a = np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32)
    axis = [1]
    dtype = np.int32
    keepdims = True
    initial = np.array(2, dtype=np.int32)
    where = np.random.choice([True, False], size=(2, 2, 2))
    promote_integers = False
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 3: 1D float32 array, axis [0], keepdims False
    a = np.random.randn(5).astype(np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.5, dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 4: 4D int32 array, axis [0, 2], keepdims True
    a = np.random.randint(1, 5, size=(2, 3, 2, 3)).astype(np.int32)
    axis = [0, 2]
    dtype = np.int32
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.ones((2, 3, 2, 3), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 5: 2D float32 array, negative values, axis [1], keepdims False
    a = np.random.randn(4, 5).astype(np.float32)
    axis = [1]
    dtype = np.float32
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.ones((4, 5), dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 6: 1D int32 array, axis [-1], keepdims True
    a = np.array([1, -2, 3]).astype(np.int32)
    axis = [-1]
    dtype = np.int32
    keepdims = True
    initial = np.array(3, dtype=np.int32)
    where = np.array([True, False, True], dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 7: 3D float32 array, axis [0, 1, 2], keepdims False
    a = np.random.randn(2, 2, 2).astype(np.float32)
    axis = [0, 1, 2]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 2), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 8: 2D int32 array, axis [0], keepdims True
    a = np.random.randint(-3, 3, size=(3, 2)).astype(np.int32)
    axis = [0]
    dtype = np.int32
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.array([[True, True], [False, True], [True, False]], dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 9: 4D float32 array, axis [1, 3], keepdims False
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [1, 3]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    # Input 10: 2D int32 array, axis [1], keepdims True
    a = np.random.randint(1, 5, size=(3, 3)).astype(np.int32)
    axis = [1]
    dtype = np.int32
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.ones((3, 3), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        "a": a, "axis": axis, "dtype": dtype, "keepdims": keepdims,
        "initial": initial, "where": where, "promote_integers": promote_integers
    })

    return list_of_inputs

generated_inputs["jax.numpy.prod_3"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_3'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_3'], lib="jax", suffix=3)
