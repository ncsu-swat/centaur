
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def linalg_outer_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays of the same length
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: Integer arrays with negative values
    x1 = np.array([-1, 0, 2, -3], dtype=np.int32)
    x2 = np.array([5, -2, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: float64 arrays of different lengths
    x1 = np.linspace(-2.0, 2.0, 5, dtype=np.float64)
    x2 = np.linspace(0.0, 1.0, 3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Complex64 arrays
    x1 = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)
    x2 = np.array([0.0 - 1.0j, 2.5 + 1.0j, -1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Complex128 arrays
    x1 = np.array([-0.5 + 1.5j, 2.0 - 3.0j, 0.0 + 0.0j], dtype=np.complex128)
    x2 = np.array([1.0 + 0.0j, -1.0 - 2.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Single-element 1D arrays
    x1 = np.array([4.2], dtype=np.float32)
    x2 = np.array([-1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Boolean arrays
    x1 = np.array([True, False, True], dtype=np.bool_)
    x2 = np.array([False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: int64 arrays of larger size
    x1 = np.arange(10, dtype=np.int64)
    x2 = np.arange(-5, 5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Empty 1D arrays
    x1 = np.array([], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: uint8 arrays
    x1 = np.array([0, 128, 255], dtype=np.uint8)
    x2 = np.array([1, 2, 4], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: int16 arrays
    x1 = np.array([-300, 0, 300], dtype=np.int16)
    x2 = np.array([10, -20], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.outer"] = linalg_outer_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.outer'.")


check_valid('jax.numpy.linalg.outer', generated_inputs['jax.numpy.linalg.outer'], lib="jax", suffix=0)
