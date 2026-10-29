
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polysub_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays, same length
    a1 = np.array([2, -3, 5], dtype=np.int32)
    a2 = np.array([1, 4, -2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 2: 1D float32 arrays, a1 shorter than a2
    a1 = np.array([3.5, -1.2], dtype=np.float32)
    a2 = np.array([2.0, 0.5, -4.1, 7.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 3: 1D float64 arrays, a1 longer than a2
    a1 = np.array([1.0, -2.5, 3.2, 0.0, -4.8], dtype=np.float64)
    a2 = np.array([5.1, -2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 4: 1D arrays of size 1 (scalar-like arrays)
    a1 = np.array([15.0], dtype=np.float32)
    a2 = np.array([-7.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 5: 2D arrays with identical shape
    a1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    a2 = np.array([[-1.0, 0.0, 2.0], [3.0, -2.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 6: 2D arrays with broadcasting along first dimension (docs example)
    a1 = np.array([[2, 3, 1]], dtype=np.int64)
    a2 = np.array([[5, 7, 3], [8, 2, 6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 7: 2D minuend and 1D subtrahend of length 1 (docs example)
    a1 = np.array([[5, 7, 9], [8, 6, 4]], dtype=np.int32)
    a2 = np.array([2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 8: 1D complex arrays
    a1 = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    a2 = np.array([2.0 - 1.0j, 1.0 + 3.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 9: 1D arrays with zeros
    a1 = np.zeros(6, dtype=np.float32)
    a2 = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 10: 1D large polynomial arrays
    a1 = np.linspace(-10.0, 10.0, 20, dtype=np.float64)
    a2 = np.linspace(5.0, -5.0, 15, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 11: 1D float16 arrays
    a1 = np.array([0.5, -1.5, 2.5], dtype=np.float16)
    a2 = np.array([-0.5, 1.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    return list_of_inputs


generated_inputs["jax.numpy.polysub"] = polysub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polysub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polysub'.")


check_valid('jax.numpy.polysub', generated_inputs['jax.numpy.polysub'], lib="jax", suffix=0)
