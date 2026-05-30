
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def convolve_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, mode='full', precision='default'
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same size, mode='same', negative values, precision='high'
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: mode='valid', float64, precision='highest'
    a = np.random.randn(10).astype(np.float64)
    v = np.random.randn(3).astype(np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex input, float32 accumulation, precision='default'
    a = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    v = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: integer input, mode='same', float accumulation, precision='default'
    a = np.array([1, -1, 2, -2, 3], dtype=np.int32)
    v = np.array([1, 0, -1], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: larger size input, mode='valid', precision='default'
    a = np.random.randn(100).astype(np.float32)
    v = np.random.randn(50).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128 accumulation, precision='high'
    a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex128)
    v = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small arrays, len(a) < len(v), precision='default'
    a = np.array([2.0, 3.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: mode='same', single element kernel 'v', precision='highest'
    a = np.random.randn(15).astype(np.float32)
    v = np.array([2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 input, mode='valid', precision='high'
    a = np.random.randn(20).astype(np.float64)
    v = np.random.randn(20).astype(np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.convolve_1"] = convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.convolve_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.convolve_1'.")


check_valid('jax.numpy.convolve', generated_inputs['jax.numpy.convolve_1'], lib="jax", suffix=1)
