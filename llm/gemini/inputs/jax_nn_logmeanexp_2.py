
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logmeanexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reduce along axis 0, keepdims=False
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    where = np.array([True, True, True, False], dtype=bool)
    axis = (0,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 2: 2D array, reduce along axis 1, keepdims=True
    x = np.random.randn(3, 4).astype(np.float32)
    where = np.ones((3, 4), dtype=bool)
    axis = (1,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 3: 2D array, reduce along both axes (0, 1), keepdims=False
    x = np.random.randn(5, 5).astype(np.float64)
    where = np.random.choice([True, False], size=(5, 5), p=[0.8, 0.2])
    where[0, 0] = True
    axis = (0, 1)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 4: 3D array, negative values, reduce along axis (0, 2), keepdims=True
    x = -np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    axis = (0, 2)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 5: 3D array, reduce along negative axis (-1,), keepdims=False
    x = np.random.randn(4, 3, 2).astype(np.float32)
    where = np.ones((4, 3, 2), dtype=bool)
    axis = (-1,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 6: 4D array, reduce along axes (1, 2), keepdims=False
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    where = np.ones((2, 3, 4, 5), dtype=bool)
    axis = (1, 2)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 7: 2D array with large values to test numerical stability, axis (0,), keepdims=True
    x = np.array([[1000.0, 2000.0], [1000.5, 1999.5]], dtype=np.float64)
    where = np.array([[True, True], [True, False]], dtype=bool)
    axis = (0,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 8: 1D array with single element, axis (0,), keepdims=True
    x = np.array([42.0], dtype=np.float32)
    where = np.array([True], dtype=bool)
    axis = (0,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 9: 3D array float64, reduce along axis (0,), keepdims=False
    x = np.random.uniform(-5.0, 5.0, size=(4, 2, 3)).astype(np.float64)
    where = np.ones((4, 2, 3), dtype=bool)
    axis = (0,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 10: 4D array, reduce along all axes (0, 1, 2, 3), keepdims=True
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    axis = (0, 1, 2, 3)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 11: 3D array with broadcastable 1D where mask, axis (1,), keepdims=False
    x = np.random.randn(2, 4, 3).astype(np.float32)
    where = np.array([[[True], [True], [False], [True]]], dtype=bool)
    axis = (1,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    return list_of_inputs


generated_inputs["jax.nn.logmeanexp_2"] = logmeanexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logmeanexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logmeanexp_2'.")


check_valid('jax.nn.logmeanexp', generated_inputs['jax.nn.logmeanexp_2'], lib="jax", suffix=2)
