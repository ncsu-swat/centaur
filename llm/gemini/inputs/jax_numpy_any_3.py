
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_any_inputs():
    list_of_inputs = []

    # Input 1: Small 2D boolean array, reduce along axis 0, keepdims=True
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis = [0]
    keepdims = True
    where = np.array([[True, True], [True, True]], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 2: 3D float array with negative values, reduce along axes [0, 2], keepdims=False
    a = np.array([[[1.0, -2.0], [0.0, 3.0]], [[0.0, 0.0], [4.0, 0.0]]], dtype=np.float32)
    axis = [0, 2]
    keepdims = False
    where = np.ones_like(a, dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 3: 1D integer array, reduce along axis 0, keepdims=True
    a = np.array([0, 1, 0, 2, 0], dtype=np.int32)
    axis = [0]
    keepdims = True
    where = np.array([True, False, True, False, True], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 4: 4D boolean array, reduce along axes [1, 2], keepdims=False
    a = np.random.choice([True, False], size=(2, 3, 2, 2))
    axis = [1, 2]
    keepdims = False
    where = np.random.choice([True, False], size=(2, 3, 2, 2))
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 5: 2D float array, broadcasted 'where' array, reduce along axis 1, keepdims=True
    a = np.array([[1.5, -0.5], [0.0, 2.3], [1.1, 0.0]], dtype=np.float64)
    axis = [1]
    keepdims = True
    where = np.array([[True], [False], [True]], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 6: 3D array, empty axis list, keepdims=False
    a = np.array([[[True]]], dtype=np.bool_)
    axis = []
    keepdims = False
    where = np.array([[[True]]], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 7: 2D int16 array, reduce along axis 1, keepdims=False
    a = np.array([[-1, 0], [0, 5]], dtype=np.int16)
    axis = [1]
    keepdims = False
    where = np.array([[True, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 8: Large 2D float64 array, reduce along axis 0, keepdims=True
    a = np.zeros((10, 10), dtype=np.float64)
    a[5, 5] = 1.0
    axis = [0]
    keepdims = True
    where = np.ones((10, 10), dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 9: 3D uint8 array, reduce along axes [0, 1], keepdims=False
    a = np.array([[[0], [0]], [[1], [0]]], dtype=np.uint8)
    axis = [0, 1]
    keepdims = False
    where = np.ones((2, 2, 1), dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    # Input 10: 1D boolean array with no True elements, reduce along axis 0, keepdims=True
    a = np.array([False, False, False], dtype=np.bool_)
    axis = [0]
    keepdims = True
    where = np.array([True, True, True], dtype=np.bool_)
    list_of_inputs.append({
        "a": copy.deepcopy(a),
        "axis": copy.deepcopy(axis),
        "keepdims": keepdims,
        "where": copy.deepcopy(where)
    })

    return list_of_inputs

generated_inputs["jax.numpy.any_3"] = jax_numpy_any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_3'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_3'], lib="jax", suffix=3)
