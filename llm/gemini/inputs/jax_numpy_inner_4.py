
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy

# Monkeypatch jax.lax.Precision to support comparison operators for numpy's min/max
try:
    jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if hasattr(other, 'value') else NotImplemented
    jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if hasattr(other, 'value') else NotImplemented
    jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if hasattr(other, 'value') else NotImplemented
    jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if hasattr(other, 'value') else NotImplemented
except Exception:
    pass

def inner_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D inputs with different batch dimensions
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float64 inputs
    a = np.array([-1.5, 2.3, -4.1, 0.5], dtype=np.float64)
    b = np.array([0.2, -3.1, -1.1, 2.2], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex numbers
    a = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    b = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "complex64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D and 2D arrays
    a = np.random.randn(2, 2, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs
    a = np.random.randint(-10, 10, size=(6,)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(6,)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "int32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D arrays
    a = np.random.randn(2, 1, 3, 5).astype(np.float32)
    b = np.random.randn(3, 2, 1, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element vectors
    a = np.array([3.14]).astype(np.float32)
    b = np.array([-1.59]).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed sizes in 1D
    a = np.random.randn(8).astype(np.float64)
    b = np.random.randn(8).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "float64"
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
