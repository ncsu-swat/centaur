
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.lax.Precision to support comparison operators for numpy's min/max
try:
    jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value
    jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value
    jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value
    jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value
except Exception:
    pass

def dot_inputs():
    list_of_inputs = []

    # Input 1: 1D vectors with default precision
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrices with high precision
    a = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 matrices with highest precision
    a = np.random.randn(3, 2).astype(np.float64)
    b = np.random.randn(2, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer matrices
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor and 1D vector
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D vector and 2D matrix
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex matrices
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    b = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed precision settings
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions
    a = np.random.randn(10, 20).astype(np.float64)
    b = np.random.randn(20, 10).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Binary-like integer matrices
    a = np.random.randint(0, 2, size=(3, 3)).astype(np.int32)
    b = np.random.randint(0, 2, size=(3, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.dot_2"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dot_2'.")


check_valid('jax.numpy.dot', generated_inputs['jax.numpy.dot_2'], lib="jax", suffix=2)
