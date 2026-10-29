
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def std_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, keepdims=False
    a = np.array([1.0, -2.5, 3.2, 4.8, -1.1, 0.0, 5.5], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 0.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 2: 2D array, float64, axis=(1,), keepdims=True
    a = np.array(
        [[2.0, 4.0, 4.0, 4.0], [5.0, 5.0, 7.0, 9.0]], dtype=np.float64
    )
    axis = (1,)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 1.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 3: 2D array, axis=(0,), negative values, with partial where mask
    a = np.array(
        [[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0], [-7.0, 8.0, -9.0]],
        dtype=np.float32,
    )
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 0.5
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 4: 3D array, axis=(0, 2), float64, keepdims=True
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4)) - 10.0
    axis = (0, 2)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 1.5
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 5: 3D array, axis=(-1,), float32, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = (-1,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 0.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 6: 4D array, axis=(1, 3), float32, keepdims=True
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 4)).astype(np.float32)
    axis = (1, 3)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 1.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 7: 2D array, axis=(0, 1), float64, reduce all axes
    a = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float64,
    )
    axis = (0, 1)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 0.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 8: 1D array, large size, negative correction
    a = np.linspace(-100.0, 100.0, 50, dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = True
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 0.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 9: 3D array, axis=(1,), float64, correction=2.0
    a = np.random.randn(4, 8, 3).astype(np.float64)
    axis = (1,)
    dtype = np.dtype(np.float64)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 2.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    # Input 10: 4D array, axis=(0, 1, 2), float32, keepdims=False
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = (0, 1, 2)
    dtype = np.dtype(np.float32)
    ddof = 0
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    mean = np.mean(a, axis=axis, keepdims=True)
    correction = 1.0
    list_of_inputs.append(
        copy.deepcopy(
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
    )

    return list_of_inputs


generated_inputs["jax.numpy.std_2"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_2'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_2'], lib="jax", suffix=2)
