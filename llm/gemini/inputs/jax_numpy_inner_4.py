
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array(list(self), dtype=object)


def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([4.0, 0.5, -1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays
    a = np.arange(6, dtype=np.float64).reshape((2, 3))
    b = np.arange(-3, 3, dtype=np.float64).reshape((2, 3))
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D vs 2D arrays
    a = np.array([-1.5, 2.5, 0.0, 4.0], dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays with mixed precision
    a = np.random.randn(2, 4, 5).astype(np.float32)
    b = np.random.randn(3, 1, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex arrays
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    b = np.array([0.0 - 1.0j, 2.0 + 2.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "high")),
        "preferred_element_type": "complex64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer arrays accumulating into float32
    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    b = np.array([[7, 8, 9]], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D and 2D arrays
    a = np.random.randn(2, 1, 3, 4).astype(np.float32)
    b = np.random.randn(5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float16 inputs upcasting to float32
    a = np.array([[-0.5, 1.5], [2.0, -3.0]], dtype=np.float16)
    b = np.array([[1.0, 0.5], [-1.0, -2.0]], dtype=np.float16)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element contraction dimension
    a = np.array([[[-5.0]], [[3.0]]], dtype=np.float64)
    b = np.array([[2.0], [-1.0], [0.5]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "highest")),
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional array contraction
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    b = np.full((3, 2), -1.0, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "high")),
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.inner_4"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.inner_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.inner_4'.")


check_valid('jax.numpy.inner', generated_inputs['jax.numpy.inner_4'], lib="jax", suffix=4)
