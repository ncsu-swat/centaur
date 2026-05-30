
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def any_inputs():
    list_of_inputs = []

    # Input 1: 2D boolean array, reduced along axis 0, keepdims=True
    a = np.array([[True, False, True], [False, False, True], [True, True, False]], dtype=bool)
    axis = np.array(0, dtype=np.int32)
    keepdims = True
    where = np.array([[True, True, False], [False, True, True], [True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 2: 2D integer array, reduced along axis 1, keepdims=False
    a = np.array([[0, 1, -1, 0], [3, 0, 0, 2]], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    keepdims = False
    where = np.array([[True, False, True, False], [True, True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 3: 3D float array, reduced along axes (0, 2), keepdims=True
    a = np.random.randn(2, 3, 2).astype(np.float32)
    axis = np.array([0, 2], dtype=np.int32)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 2))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 4: 1D boolean array, reduced along axis 0, keepdims=False
    a = np.array([False, True, False, False, True], dtype=bool)
    axis = np.array(0, dtype=np.int32)
    keepdims = False
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 5: 2D float64 array, reduced along axis 1, keepdims=True
    a = np.random.randn(4, 4).astype(np.float64)
    axis = np.array(1, dtype=np.int32)
    keepdims = True
    where = np.ones((4, 4), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 6: 3D int array, reduced along axis 1, keepdims=False
    a = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=np.int32)
    axis = np.array([1], dtype=np.int32)
    keepdims = False
    where = np.array([[[True, False], [True, True]], [[False, True], [True, False]]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 7: 3D boolean array, reduced along axis 2, keepdims=True
    a = np.random.choice([True, False], size=(3, 1, 4))
    axis = np.array(2, dtype=np.int32)
    keepdims = True
    where = np.random.choice([True, False], size=(3, 1, 4))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 8: 2D int array, reduced along axes (0, 1), keepdims=False
    a = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.int32)
    axis = np.array([0, 1], dtype=np.int32)
    keepdims = False
    where = np.array([[True, True], [False, False], [True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 9: 2D float array with shape (1, 5), reduced along axis 0, keepdims=True
    a = np.random.randn(1, 5).astype(np.float32)
    axis = np.array(0, dtype=np.int32)
    keepdims = True
    where = np.array([[True, False, True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 10: 4D boolean array, reduced along axes (0, 2, 3), keepdims=False
    a = np.random.choice([True, False], size=(2, 3, 2, 2))
    axis = np.array([0, 2, 3], dtype=np.int32)
    keepdims = False
    where = np.random.choice([True, False], size=(2, 3, 2, 2))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.any_4"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_4'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_4'], lib="jax", suffix=4)
