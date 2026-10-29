
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def std_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, float32
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.array([True, True, True, True, True], dtype=bool)
    mean = np.array([3.0], dtype=np.float32)
    correction = 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 2: 2D array, axis=0, keepdims=True, float64, correction=1
    a = np.arange(12, dtype=np.float64).reshape(4, 3)
    axis = 0
    dtype = np.float64
    ddof = 0
    keepdims = True
    where = np.ones((4, 3), dtype=bool)
    mean = np.mean(a, axis=0, keepdims=True)
    correction = 1
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 3: 2D array, axis=1, keepdims=False, with some masked where values
    a = np.array(
        [[2.5, -1.0, 3.2], [0.0, 4.5, -2.1], [-3.3, 1.2, 5.0]],
        dtype=np.float32,
    )
    axis = 1
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.array(
        [[True, True, True], [True, True, True], [True, True, True]], dtype=bool
    )
    mean = np.mean(a, axis=1, keepdims=True).astype(np.float32)
    correction = 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 4: 3D array, axis=-1, keepdims=True, negative values
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    axis = -1
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.ones((2, 3, 4), dtype=bool)
    mean = np.mean(a, axis=-1, keepdims=True)
    correction = 1
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 5: 3D array, axis=1, complex64
    a = (
        np.random.randn(3, 4, 2).astype(np.float32)
        + 1j * np.random.randn(3, 4, 2).astype(np.float32)
    ).astype(np.complex64)
    axis = 1
    dtype = np.complex64
    ddof = 0
    keepdims = False
    where = np.ones((3, 4, 2), dtype=bool)
    mean = np.mean(a, axis=1, keepdims=True)
    correction = 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 6: 4D array, axis=2, float64, correction=2
    a = np.random.randn(2, 2, 5, 3).astype(np.float64)
    axis = 2
    dtype = np.float64
    ddof = 0
    keepdims = True
    where = np.ones((2, 2, 5, 3), dtype=bool)
    mean = np.mean(a, axis=2, keepdims=True)
    correction = 2
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 7: 2D integer input, output float32, axis=0
    a = np.array(
        [[10, 20, 30], [-10, -20, -30], [5, 15, 25]], dtype=np.int32
    )
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.ones((3, 3), dtype=bool)
    mean = np.mean(a, axis=0, keepdims=True).astype(np.float32)
    correction = 1
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 8: 1D array with where mask excluding an element
    a = np.array([10.0, -10.0, 50.0, 20.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = True
    where = np.array([True, True, False, True], dtype=bool)
    mean = np.array([20.0 / 3.0], dtype=np.float32)
    correction = 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 9: 3D array, axis=0, keepdims=False, float16 cast to float32
    a = np.ones((3, 2, 2), dtype=np.float16) * 4.0
    axis = 0
    dtype = np.float32
    ddof = 0
    keepdims = False
    where = np.ones((3, 2, 2), dtype=bool)
    mean = np.full((1, 2, 2), 4.0, dtype=np.float32)
    correction = 0
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    # Input 10: 2D array, axis=-2, float64
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    axis = -2
    dtype = np.float64
    ddof = 0
    keepdims = False
    where = np.ones((3, 2), dtype=bool)
    mean = np.mean(a, axis=-2, keepdims=True)
    correction = 1
    list_of_inputs.append(
        {
            "a": a,
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean,
            "correction": correction,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.std_1"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_1'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_1'], lib="jax", suffix=1)
