
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Patch jax.lax.Precision to support comparison operators for numpy's min/max reduction
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if isinstance(other, jax.lax.Precision) else NotImplemented

def convolve_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, mode='full'
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different sizes, mode='same', float64
    a = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0], dtype=np.float64)
    v = np.array([1.0, 2.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mode='valid', float32
    a = np.arange(10, dtype=np.float32)
    v = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex inputs, mode='full'
    a = np.array([1 + 1j, 2 - 2j, 3 + 3j], dtype=np.complex64)
    v = np.array([0.5 + 0.5j, 1.0], dtype=np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 inputs with float64 element type preference
    a = np.random.randn(8).astype(np.float64)
    v = np.random.randn(4).astype(np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs (stored as float/int in tensor), mode='valid'
    a = np.array([1, 3, 5, 7, 9], dtype=np.int32)
    v = np.array([2, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small arrays, mode='same'
    a = np.array([5.0], dtype=np.float32)
    v = np.array([2.0, -1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros, mode='full'
    a = np.zeros(20, dtype=np.float32)
    v = np.zeros(5, dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays, mode='valid'
    a = np.random.randn(100).astype(np.float32)
    v = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive values mixed, mode='same'
    a = np.array([-1.5, 2.5, -3.5, 4.5, -5.5], dtype=np.float64)
    v = np.array([0.1, -0.2, 0.1], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.convolve_2"] = convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.convolve_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.convolve_2'.")


check_valid('jax.numpy.convolve', generated_inputs['jax.numpy.convolve_2'], lib="jax", suffix=2)
