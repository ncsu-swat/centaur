
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=dtype or np.int32)


def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([4.0, 0.5, -1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 matrices with matching inner dimension
    a = np.random.randn(2, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 matrices with negative and positive values
    a = np.random.uniform(-5.0, 5.0, size=(5, 3)).astype(np.float64)
    b = np.random.uniform(-5.0, 5.0, size=(4, 3)).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 vectors
    a = np.array([-1, 2, -3, 4, 5], dtype=np.int32)
    b = np.array([10, -2, 0, 3, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "high")),
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D and 2D float32 arrays
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 arrays
    a = np.random.randn(2, 3, 6).astype(np.float32)
    b = np.random.randn(4, 2, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "highest")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 1D arrays
    a = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    b = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "default")),
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D and 3D float32 arrays
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("high", "highest")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 with higher precision
    a = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    b = np.linspace(0.0, 2.0, 10, dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("highest", "high")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int64 arrays
    a = np.random.randint(-10, 10, size=(3, 4), dtype=np.int64)
    b = np.random.randint(-10, 10, size=(5, 4), dtype=np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.inner_2"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.inner_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.inner_2'.")


check_valid('jax.numpy.inner', generated_inputs['jax.numpy.inner_2'], lib="jax", suffix=2)
