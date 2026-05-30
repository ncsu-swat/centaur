
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def all_inputs():
    list_of_inputs = []

    # Input 1: 2D array, axis=(0,), keepdims=False, where same shape
    a = np.random.choice([True, False], size=(3, 4))
    axis = (0,)
    keepdims = False
    where = np.random.choice([True, False], size=(3, 4))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 2: 3D array, axis=(0, 2), keepdims=True, where same shape
    a = np.random.choice([True, False], size=(2, 3, 4))
    axis = (0, 2)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 3: 1D array, axis=(0,), keepdims=False, where same shape
    a = np.random.choice([True, False], size=(5,))
    axis = (0,)
    keepdims = False
    where = np.random.choice([True, False], size=(5,))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 4: 2D array of integers, axis=(1,), keepdims=True, where broadcastable
    a = np.random.randint(-2, 3, size=(3, 3))
    axis = (1,)
    keepdims = True
    where = np.random.choice([True, False], size=(3, 1))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 5: 4D array of floats, axis=(1, 3), keepdims=False, where same shape
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    keepdims = False
    where = np.random.choice([True, False], size=(2, 2, 2, 2))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 6: 2D array, empty axis tuple, keepdims=True, where broadcastable
    a = np.random.choice([True, False], size=(4, 4))
    axis = ()
    keepdims = True
    where = np.random.choice([True, False], size=(1, 4))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 7: 3D array, axis=(1, 2), keepdims=False, where same shape
    a = np.random.choice([True, False], size=(3, 3, 3))
    axis = (1, 2)
    keepdims = False
    where = np.random.choice([True, False], size=(3, 3, 3))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 8: 2D array of integers, axis=(0,), keepdims=True, where same shape
    a = np.random.randint(0, 2, size=(5, 2))
    axis = (0,)
    keepdims = True
    where = np.random.choice([True, False], size=(5, 2))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 9: 1D array, axis=(0,), keepdims=True, where broadcastable (single element)
    a = np.random.choice([True, False], size=(10,))
    axis = (0,)
    keepdims = True
    where = np.array([True])
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    # Input 10: 5D array, axis=(0, 2, 4), keepdims=False, where same shape
    a = np.random.choice([True, False], size=(2, 2, 2, 2, 2))
    axis = (0, 2, 4)
    keepdims = False
    where = np.random.choice([True, False], size=(2, 2, 2, 2, 2))
    list_of_inputs.append({"a": a, "axis": axis, "keepdims": keepdims, "where": where})

    return list_of_inputs

generated_inputs["jax.numpy.all_2"] = all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_2'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_2'], lib="jax", suffix=2)
