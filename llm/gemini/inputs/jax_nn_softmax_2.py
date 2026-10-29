
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=(-1,)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (-1,)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 2: 2D array, axis=(1,)
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.ones((3, 4), dtype=bool)
    where[0, 2] = False
    where[1, 0] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 3: 2D array, float64, axis=(0,)
    x = np.random.randn(4, 5).astype(np.float64)
    axis = (0,)
    where = np.ones((4, 5), dtype=bool)
    where[2, :] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 4: 3D array, axis=(-1,)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (-1,)
    where = np.random.rand(2, 3, 4) > 0.2
    where[:, :, 0] = True  # Ensure at least one True per slice
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 5: 3D array, multi-axis=(1, 2)
    x = np.random.randn(2, 4, 4).astype(np.float32)
    axis = (1, 2)
    where = np.ones((2, 4, 4), dtype=bool)
    where[:, 0, 0] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 6: 4D array, axis=(-2,)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (-2,)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    where[0, 1, 2, :] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 7: 2D array with negative values, axis=(0, 1)
    x = -np.random.exponential(scale=2.0, size=(5, 5)).astype(np.float32)
    axis = (0, 1)
    where = np.ones((5, 5), dtype=bool)
    where[0, 0] = False
    where[4, 4] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 8: 1D array, axis=(0,), all True mask
    x = np.linspace(-10.0, 10.0, 20, dtype=np.float64)
    axis = (0,)
    where = np.ones(20, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 9: 3D array with broadcasting mask, axis=(-1,)
    x = np.random.randn(4, 3, 6).astype(np.float32)
    axis = (-1,)
    where = np.array([True, True, False, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 10: 4D array, axis=(2, 3)
    x = np.random.randn(2, 3, 4, 5).astype(np.float64)
    axis = (2, 3)
    where = np.ones((2, 3, 4, 5), dtype=bool)
    where[:, :, 1, 2] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    # Input 11: 3D array, axis=(0, 2)
    x = np.random.uniform(-5.0, 5.0, size=(3, 4, 3)).astype(np.float32)
    axis = (0, 2)
    where = np.ones((3, 4, 3), dtype=bool)
    where[1, :, 1] = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.nn.softmax_2"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softmax_2'.")


check_valid('jax.nn.softmax', generated_inputs['jax.nn.softmax_2'], lib="jax", suffix=2)
