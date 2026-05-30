
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_any_inputs():
    list_of_inputs = []

    # Input 1: 2D bool array, axis=(0,), keepdims=False
    a = np.array([[True, False, True], [False, False, False]], dtype=bool)
    axis = (0,)
    keepdims = False
    where = np.array([[True, True, True], [True, True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 2: 2D int array, axis=(1,), keepdims=True
    a = np.array([[0, 1, 0], [0, 0, 0], [2, 3, 0]], dtype=np.int32)
    axis = (1,)
    keepdims = True
    where = np.array([[True, True, False], [True, True, True], [True, False, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 3: 3D float array, axis=(0, 2), keepdims=False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    keepdims = False
    where = (a > 0.0)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 4: 2D bool array, axis=(0, 1), keepdims=True
    a = np.random.choice([True, False], size=(5, 5))
    axis = (0, 1)
    keepdims = True
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 5: 3D float64 array, axis=(1,), keepdims=False
    a = np.random.randn(3, 1, 4).astype(np.float64)
    axis = (1,)
    keepdims = False
    where = np.ones((3, 1, 4), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 6: 3D bool array, axis=(0, 1, 2), keepdims=True
    a = np.array([[[False]]], dtype=bool)
    axis = (0, 1, 2)
    keepdims = True
    where = np.array([[[True]]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 7: 2D int64 array with negative elements, axis=(0,), keepdims=False
    a = np.array([[-1, 0], [0, 2], [3, -4]], dtype=np.int64)
    axis = (0,)
    keepdims = False
    where = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 8: 3D bool array, axis=(2,), keepdims=True
    a = np.random.choice([True, False], size=(4, 4, 4))
    axis = (2,)
    keepdims = True
    where = np.random.choice([True, False], size=(4, 4, 4))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 9: 4D float32 array, axis=(1, 3), keepdims=False
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    keepdims = False
    where = (a < 0.5)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 10: 2D bool array with all False, axis=(0, 1), keepdims=False
    a = np.zeros((3, 4), dtype=bool)
    axis = (0, 1)
    keepdims = False
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.any_2"] = jax_numpy_any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_2'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_2'], lib="jax", suffix=2)
