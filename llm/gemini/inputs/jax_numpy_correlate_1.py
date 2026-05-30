
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_correlate_inputs():
    list_of_inputs = []

    # Input 1: Basic real inputs, mode='valid'
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different lengths, mode='full', precision='high'
    a = np.array([-1.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    v = np.array([0.5, -0.5, 1.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 precision, mode='same', precision='highest'
    a = np.random.randn(10).astype(np.float64)
    v = np.random.randn(5).astype(np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex inputs, complex64, mode='full'
    a = (np.random.randn(6) + 1j * np.random.randn(6)).astype(np.complex64)
    v = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex inputs, complex128, mode='valid'
    a = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex128)
    v = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large inputs, negative values, mode='same'
    a = np.random.uniform(-10, 10, 100).astype(np.float32)
    v = np.random.uniform(-5, 5, 20).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer inputs with preferred_element_type as np.float32
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    v = np.array([2, -1, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small sizes, length 1
    a = np.array([2.5], dtype=np.float32)
    v = np.array([-1.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 'a' longer than 'v', mode='full', precision='high'
    a = np.random.randn(15).astype(np.float32)
    v = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 'a' longer than 'v', mode='same', precision='highest'
    a = np.random.randn(20).astype(np.float32)
    v = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.correlate_1"] = jax_numpy_correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.correlate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.correlate_1'.")


check_valid('jax.numpy.correlate', generated_inputs['jax.numpy.correlate_1'], lib="jax", suffix=1)
