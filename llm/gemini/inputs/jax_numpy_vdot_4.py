
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=np.int32)


def vdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors
    a = np.array([1.0, -2.0, 3.5, -4.0, 5.0], dtype=np.float32)
    b = np.array([0.5, 2.0, -1.0, 3.0, -0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex64 vectors
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    b = np.array([2.0 - 1.0j, 1.0 + 3.0j, -0.5 - 2.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": "complex64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array and 1D vector of same size, float64
    a = np.arange(6, dtype=np.float64).reshape((2, 3)) - 3.0
    b = np.array([1.5, -0.5, 2.0, -1.0, 0.0, 3.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D and 2D arrays with matching size
    a = np.ones((2, 2, 2), dtype=np.float32)
    b = np.full((4, 2), -2.0, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "high")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int32 arrays with negative values
    a = np.array([-10, 20, -30, 40, -50], dtype=np.int32)
    b = np.array([2, -3, 4, -5, 6], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "int32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex128 vectors
    a = np.array([-1.5 + 4.0j, 2.5 - 3.0j, 0.0 + 1.0j], dtype=np.complex128)
    b = np.array([3.0 - 2.0j, -1.0 + 1.0j, 2.0 + 0.0j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "default")),
        "preferred_element_type": "complex128",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 matrices with matching sizes
    a = np.array(
        [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]],
        dtype=np.float64,
    )
    b = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "default")),
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor and 1D vector flattened
    a = np.random.randn(2, 1, 2, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D scalar-like arrays
    a = np.array(3.14, dtype=np.float32)
    b = np.array(-2.71, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 large size with high precision
    a = np.linspace(-5.0, 5.0, 20, dtype=np.float32)
    b = np.linspace(1.0, 10.0, 20, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "highest")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D complex64 matrices with different shapes but same size
    a = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    b = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "high")),
        "preferred_element_type": "complex64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.vdot_4"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_4'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_4'], lib="jax", suffix=4)
