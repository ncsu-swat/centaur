
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Add comparison operators to jax.lax.Precision so that numpy min/max operations work on them
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if isinstance(other, jax.lax.Precision) else NotImplemented

def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32, high precision
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D and 2D arrays, float64, negative values, highest precision
    a = -np.random.rand(4).astype(np.float64)
    b = np.random.randn(3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer arrays
    a = np.random.randint(-10, 10, size=(2, 3)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(5, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimensional inputs (3D and 2D)
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex inputs
    a = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    b = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH),
        "preferred_element_type": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 inputs with float32 accumulation
    a = np.random.randn(2, 6).astype(np.float16)
    b = np.random.randn(1, 6).astype(np.float16)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large dimension contraction
    a = np.random.randn(128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64 arrays, float64 accumulation
    a = np.random.randint(-5, 5, size=(3, 2)).astype(np.int64)
    b = np.random.randint(-5, 5, size=(3, 2)).astype(np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D and 3D inputs
    a = np.random.randn(2, 2, 2, 3).astype(np.float32)
    b = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float32
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
