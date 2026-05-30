
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.lax.Precision to support comparison so np.min/np.max do not fail on object arrays
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if hasattr(other, 'value') else NotImplemented

def vdot_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D vectors
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 2D arrays (flattened size 6)
    a = np.random.randn(2, 3).astype(np.float64)
    b = np.random.randn(2, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64 1D vectors
    a = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    b = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype("complex64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int32 1D vectors with negative values
    a = np.array([-1, 2, -3, 4], dtype=np.int32)
    b = np.array([5, -6, 7, -8], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32 3D tensors (flattened size 8)
    a = np.random.randn(2, 2, 2).astype(np.float32)
    b = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 1D vectors
    a = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex128)
    b = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype("complex128")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 1D vectors
    a = np.random.randn(8).astype(np.float16)
    b = np.random.randn(8).astype(np.float16)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int64 2D arrays (flattened size 9)
    a = np.random.randint(-10, 10, size=(3, 3)).astype(np.int64)
    b = np.random.randint(-10, 10, size=(3, 3)).astype(np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex64 2D arrays (flattened size 8)
    a = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex64)
    b = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype("complex64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 3D arrays (flattened size 10)
    a = np.random.randn(1, 5, 2).astype(np.float64)
    b = np.random.randn(1, 5, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vdot_2"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_2'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_2'], lib="jax", suffix=2)
