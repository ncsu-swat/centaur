
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array(list(self), dtype=object)


def convolve_inputs():
    list_of_inputs = []

    # Input 1: Float32 inputs with full mode
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 inputs with same mode
    a = np.linspace(-2.0, 2.0, 10, dtype=np.float64)
    v = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Equal length inputs with valid mode
    a = np.array([1.5, -2.5, 3.0, -0.5, 4.0], dtype=np.float32)
    v = np.array([2.0, 1.0, -1.0, 0.5, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 inputs with full mode
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    v = np.array([0.5 - 1.5j, 2.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32 inputs with valid mode and longer kernel
    a = np.arange(12, dtype=np.float32) - 6.0
    v = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("default", "high")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs with type promotion via preferred_element_type
    a = np.array([1, -3, 5, -7, 9], dtype=np.int32)
    v = np.array([2, -1, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128 inputs with valid mode
    a = np.array(
        [1.0 - 1.0j, 2.0 + 3.0j, -0.5 + 4.0j, 3.5 - 2.5j], dtype=np.complex128
    )
    v = np.array([0.5 + 0.5j, -1.0 + 2.0j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single-element kernel with same mode
    a = np.random.randn(15).astype(np.float32)
    v = np.array([2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("high", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Input a smaller than kernel v with full mode
    a = np.array([1.0, -1.0], dtype=np.float64)
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("default", "high")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32 inputs accumulating into float64
    a = np.linspace(-10.0, 10.0, 16, dtype=np.float32)
    v = np.array([0.2, -0.4, 0.6, -0.8, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("highest", "default")),
        "preferred_element_type": np.dtype(np.float64),
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
