
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logmeanexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce over axis 0
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True, True], dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 2: 2D array, reduce over axis 0, keepdims True
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (0,)
    where = np.ones((3, 4), dtype=bool)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 3: 2D array, reduce over axis 1, float64
    x = np.random.randn(5, 5).astype(np.float64)
    axis = (1,)
    where = np.ones((5, 5), dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 4: 3D array, reduce over multiple axes (0, 2)
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    axis = (0, 2)
    where = np.ones((2, 3, 4), dtype=bool)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 5: 3D array with negative axis (-1,)
    x = np.array([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32)
    axis = (-1,)
    where = np.array([[[True, True], [True, True]]], dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 6: 4D array, reduce over (1, 2), keepdims True
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (1, 2)
    where = np.ones((2, 3, 4, 5), dtype=bool)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 7: 2D array, reduce all axes (0, 1), keepdims True
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    axis = (0, 1)
    where = np.ones((2, 2), dtype=bool)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 8: 1D array with negative values and float64
    x = np.array([-100.0, -200.0, -300.0], dtype=np.float64)
    axis = (0,)
    where = np.array([True, True, True], dtype=bool)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 9: 3D array, single axis reduction
    x = np.random.randn(4, 3, 2).astype(np.float32)
    axis = (1,)
    where = np.ones((4, 3, 2), dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 10: 2D array with all axes reduced (0, 1), keepdims False
    x = np.linspace(-2.0, 2.0, 12, dtype=np.float32).reshape(3, 4)
    axis = (0, 1)
    where = np.ones((3, 4), dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    # Input 11: 4D small array, reduce over (2, 3)
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    axis = (2, 3)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims})
    )

    return list_of_inputs


generated_inputs["jax.nn.logmeanexp_3"] = logmeanexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logmeanexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logmeanexp_3'.")


check_valid('jax.nn.logmeanexp', generated_inputs['jax.nn.logmeanexp_3'], lib="jax", suffix=3)
