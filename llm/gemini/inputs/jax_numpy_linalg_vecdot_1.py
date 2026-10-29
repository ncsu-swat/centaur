
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vecdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, axis = -1
    x1 = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    x2 = np.array([0.5, 2.0, -1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 2: 2D float64 arrays, axis = 0
    x1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    x2 = np.array([[-1.0, 0.5], [2.0, -1.5], [0.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "high",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 3: Complex128 1D vectors with conjugate handling
    x1 = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 1.5j], dtype=np.complex128)
    x2 = np.array([2.0 - 1.0j, 1.0 + 0.0j, -2.0 - 2.0j], dtype=np.complex128)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.complex128),
        }
    )

    # Input 4: 2D arrays broadcasting (2, 4) and (1, 4), axis = -1
    x1 = np.arange(8, dtype=np.float32).reshape(2, 4)
    x2 = np.ones((1, 4), dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 5: 3D float32 arrays contracting along axis = 1
    x1 = np.random.randn(3, 5, 2).astype(np.float32)
    x2 = np.random.randn(3, 5, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 1,
            "precision": "high",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 6: Integer arrays, casting result to float64
    x1 = np.array([-5, 0, 12, 3], dtype=np.int32)
    x2 = np.array([2, -4, 1, 7], dtype=np.int32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 7: 3D with complex64 dtype, axis = 2
    x1 = (
        np.random.randn(2, 3, 4).astype(np.float32)
        + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    ).astype(np.complex64)
    x2 = (
        np.random.randn(2, 3, 4).astype(np.float32)
        + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 2,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.complex64),
        }
    )

    # Input 8: 4D arrays broadcasting along non-contracted dimensions, axis = -2
    x1 = np.ones((2, 1, 6, 3), dtype=np.float32)
    x2 = np.ones((1, 4, 6, 3), dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -2,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 9: 2D arrays with axis = -2
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -2,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 10: 4D float64 arrays, axis = 0
    x1 = np.random.randn(4, 2, 2, 2).astype(np.float64)
    x2 = np.random.randn(4, 2, 2, 2).astype(np.float64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "high",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.vecdot_1"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vecdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vecdot_1'.")


check_valid('jax.numpy.linalg.vecdot', generated_inputs['jax.numpy.linalg.vecdot_1'], lib="jax", suffix=1)
