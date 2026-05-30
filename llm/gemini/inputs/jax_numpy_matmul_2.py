
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy

# Monkeypatch jax.lax.Precision to support comparison operators for numpy min/max
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value

def matmul_inputs():
    list_of_inputs = []
    
    # Input 1: 1D vectors dot product
    list_of_inputs.append({
        "a": np.array([1.0, -2.0, 3.0], dtype=np.float32),
        "b": np.array([-4.0, 5.0, -6.0], dtype=np.float32),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('float32')
    })
    
    # Input 2: 2D standard matmul (float32)
    list_of_inputs.append({
        "a": np.random.randn(3, 4).astype(np.float32),
        "b": np.random.randn(4, 5).astype(np.float32),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype('float32')
    })

    # Input 3: float64 standard matmul
    list_of_inputs.append({
        "a": np.random.randn(2, 5).astype(np.float64),
        "b": np.random.randn(5, 3).astype(np.float64),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype('float64')
    })

    # Input 4: int32 matrices
    list_of_inputs.append({
        "a": np.random.randint(-10, 10, size=(4, 4)).astype(np.int32),
        "b": np.random.randint(-10, 10, size=(4, 2)).astype(np.int32),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('int32')
    })

    # Input 5: complex64 matrices
    list_of_inputs.append({
        "a": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64),
        "b": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('complex64')
    })

    # Input 6: Batched 3D matrices
    list_of_inputs.append({
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "b": np.random.randn(2, 4, 5).astype(np.float32),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype('float32')
    })

    # Input 7: 1D 'a' and 2D 'b'
    list_of_inputs.append({
        "a": np.random.randn(5).astype(np.float32),
        "b": np.random.randn(5, 3).astype(np.float32),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('float32')
    })

    # Input 8: 2D 'a' and 1D 'b'
    list_of_inputs.append({
        "a": np.random.randn(3, 5).astype(np.float32),
        "b": np.random.randn(5).astype(np.float32),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('float32')
    })

    # Input 9: Multi-dimensional batched with broadcasting
    list_of_inputs.append({
        "a": np.random.randn(1, 3, 4).astype(np.float32),
        "b": np.random.randn(5, 4, 2).astype(np.float32),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype('float32')
    })

    # Input 10: Larger matrix multiplication
    list_of_inputs.append({
        "a": np.random.randn(64, 128).astype(np.float32),
        "b": np.random.randn(128, 64).astype(np.float32),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype('float32')
    })

    return list_of_inputs

generated_inputs["jax.numpy.matmul_2"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matmul_2'.")


check_valid('jax.numpy.matmul', generated_inputs['jax.numpy.matmul_2'], lib="jax", suffix=2)
