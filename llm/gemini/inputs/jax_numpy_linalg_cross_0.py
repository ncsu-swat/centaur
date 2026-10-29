
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_linalg_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D vectors, float32, default-like axis=-1
    x1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x2 = np.array([-4.0, 5.0, -6.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -1})
    )

    # Input 2: 1D vectors with axis=0
    x1 = np.array([0.5, 1.5, -2.5], dtype=np.float64)
    x2 = np.array([3.0, 0.0, -1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": 0})
    )

    # Input 3: 2D arrays, axis=-1, int32
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x2 = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -1})
    )

    # Input 4: 2D arrays, axis=0 with shape (3, N)
    x1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    x2 = np.ones((3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": 0})
    )

    # Input 5: 2D with broadcasting: (1, 3) and (4, 3), axis=1
    x1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    x2 = np.random.randn(4, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": 1})
    )

    # Input 6: 3D arrays, axis=-1
    x1 = np.random.randn(2, 5, 3).astype(np.float64)
    x2 = np.random.randn(2, 5, 3).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -1})
    )

    # Input 7: 3D arrays, axis=1 with shape (2, 3, 4)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": 1})
    )

    # Input 8: 3D arrays with broadcasting along batch dim: (2, 1, 3) and (1, 4, 3)
    x1 = np.random.randn(2, 1, 3).astype(np.float32)
    x2 = np.random.randn(1, 4, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -1})
    )

    # Input 9: 4D arrays, axis=-2 with shape (2, 2, 3, 4)
    x1 = np.random.randn(2, 2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 2, 3, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -2})
    )

    # Input 10: Integer arrays with int64
    x1 = np.array([[[1, 0, 0]], [[0, 1, 0]]], dtype=np.int64)
    x2 = np.array([[[0, 1, 0]], [[0, 0, 1]]], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": -1})
    )

    # Input 11: Complex numbers, axis=0
    x1 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    x2 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"x1": x1, "x2": x2, "axis": 0})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.cross"] = jax_numpy_linalg_cross_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cross'.")


check_valid('jax.numpy.linalg.cross', generated_inputs['jax.numpy.linalg.cross'], lib="jax", suffix=0)
