
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def prod_inputs():
    list_of_inputs = []

    # Input 1: 2D integer array, axis 0, promote_integers True
    a = np.arange(1, 7, dtype=np.int32).reshape(2, 3)
    axis = 0
    dtype = np.dtype('int32')
    keepdims = False
    initial = np.array(2, dtype=np.int32)
    where = np.ones((2, 3), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 2: 2D float array, axis 1, keepdims True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    dtype = np.dtype('float32')
    keepdims = True
    initial = np.array(1.5, dtype=np.float32)
    where = np.random.choice([True, False], size=(3, 4))
    promote_integers = False
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 3: 3D int64 array, negative axis
    a = np.arange(1, 9, dtype=np.int64).reshape(2, 2, 2)
    axis = -1
    dtype = np.dtype('int64')
    keepdims = False
    initial = np.array(1, dtype=np.int64)
    where = np.ones((2, 2, 2), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 4: 1D float64 array, where mask with some False
    a = np.random.rand(5).astype(np.float64)
    axis = 0
    dtype = np.dtype('float64')
    keepdims = True
    initial = np.array(1.0, dtype=np.float64)
    where = np.array([True, False, True, False, True])
    promote_integers = False
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 5: 3D int32 array, with integer promotion and higher output dtype
    a = np.ones((2, 3, 4), dtype=np.int32)
    axis = 2
    dtype = np.dtype('int64')
    keepdims = False
    initial = np.array(3, dtype=np.int64)
    where = np.ones((2, 3, 4), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 6: 4D float32 array
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 1
    dtype = np.dtype('float32')
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 7: 2D int16 array with custom boolean where mask
    a = np.array([[2, 3], [4, 5]], dtype=np.int16)
    axis = -2
    dtype = np.dtype('int32')
    keepdims = True
    initial = np.array(2, dtype=np.int32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    promote_integers = False
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 8: 1D float64 array with all values used
    a = np.random.uniform(0.5, 2.0, size=(10,)).astype(np.float64)
    axis = 0
    dtype = np.dtype('float64')
    keepdims = False
    initial = np.array(0.5, dtype=np.float64)
    where = np.ones((10,), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 9: 2D int32 array with keepdims True
    a = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    axis = 0
    dtype = np.dtype('int32')
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.ones((3, 4), dtype=bool)
    promote_integers = True
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    # Input 10: 3D float32 array, random boolean mask
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = 2
    dtype = np.dtype('float32')
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.random.rand(3, 3, 3) > 0.5
    promote_integers = False
    list_of_inputs.append({
        'a': a, 'axis': axis, 'dtype': dtype, 'keepdims': keepdims,
        'initial': initial, 'where': where, 'promote_integers': promote_integers
    })

    return list_of_inputs

generated_inputs["jax.numpy.prod_1"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_1'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_1'], lib="jax", suffix=1)
