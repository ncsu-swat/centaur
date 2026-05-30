
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def outer_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 arrays
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 2: 1D int32 arrays with negative values
    a = np.array([-1, 0, 1, 2], dtype=np.int32)
    b = np.array([3, -4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 3: 2D arrays (will be flattened)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 4: 3D arrays
    a = np.random.randn(2, 2, 2).astype(np.float64)
    b = np.random.randn(1, 3, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 5: Boolean arrays
    a = np.array([True, False, True], dtype=bool)
    b = np.array([False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 6: Complex arrays
    a = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    b = np.array([2 - 1j, 5], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 7: One-element arrays
    a = np.array([5.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 8: Large 1D arrays
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(200).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 9: Mixed shape dimensions
    a = np.random.randn(2, 2, 1, 3).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 10: Large integers
    a = np.array([1000, 2000, 3000], dtype=np.int64)
    b = np.array([-100, 200], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    return list_of_inputs

generated_inputs["jax.numpy.outer"] = outer_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.outer'.")


check_valid('jax.numpy.outer', generated_inputs['jax.numpy.outer'], lib="jax", suffix=0)
