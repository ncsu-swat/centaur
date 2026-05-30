
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Patch Precision to make it fully comparable for numpy min/max in the test framework
try:
    jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value
    jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value
    jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value
    jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value
except Exception:
    pass

def vdot_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 1D vectors
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex64 1D vectors
    a = np.array([1j, 2j, 3j], dtype=np.complex64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": "complex64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float64 vectors
    a = np.array([-1.5, -2.5, 3.5], dtype=np.float64)
    b = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array to be flattened and 1D array
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer arrays
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "int32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean arrays
    a = np.array([True, False, True], dtype=np.bool_)
    b = np.array([True, True, False], dtype=np.bool_)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "int32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128 vectors
    a = np.array([1.0 + 2.0j, -3.0j], dtype=np.complex128)
    b = np.array([2.0 - 1.0j, 4.0], dtype=np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "complex128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array and 1D array of the same size
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(24).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Two multi-dimensional arrays of different shapes but same total size
    a = np.random.randn(2, 5).astype(np.float32)
    b = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger float64 vectors
    a = np.random.randn(100).astype(np.float64)
    b = np.random.randn(100).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": "float64"
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
