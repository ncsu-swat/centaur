
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def outer_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 2: 1D int32 arrays with negative values
    a = np.array([-1, 0, 1], dtype=np.int32)
    b = np.array([2, -3, 4, -5], dtype=np.int32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 3: Different lengths float64
    a = np.array([0.5, -1.5], dtype=np.float64)
    b = np.array([2.0, 3.0, -4.0, 5.0, 6.0], dtype=np.float64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 4: 2D arrays (flattened by outer)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 5: 3D array and 1D array
    a = np.ones((2, 3, 4), dtype=np.float32)
    b = np.array([1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 6: Complex numbers
    a = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    b = np.array([2.0 - 1.0j, 0.0 + 3.0j, 5.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 7: Boolean arrays
    a = np.array([True, False, True], dtype=bool)
    b = np.array([False, True], dtype=bool)
    list_of_inputs.append({"a": a, "b": b})

    # Input 8: Mixed 0D (scalar-like) arrays
    a = np.array(3.5, dtype=np.float32)
    b = np.array([-2.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 9: Large 1D int64 arrays
    a = np.arange(10, dtype=np.int64)
    b = np.arange(-5, 5, dtype=np.int64)
    list_of_inputs.append({"a": a, "b": b})

    # Input 10: Empty 1D arrays
    a = np.array([], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # Input 11: Multi-dimensional mixed shapes
    a = np.random.randn(3, 2).astype(np.float64)
    b = np.random.randn(2, 4, 1).astype(np.float64)
    list_of_inputs.append({"a": a, "b": b})

    return [copy.deepcopy(x) for x in list_of_inputs]


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
