
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hankel_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with matching lengths
    c = np.array([1, 2, 3, 4], dtype=np.int32)
    r = np.array([4, 5, 6, 7], dtype=np.int32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 2: 1D float32 arrays where M > N
    c = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    r = np.array([-3.0, 4.2, -5.1, 6.0, 7.8], dtype=np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 3: 1D float64 arrays where N > M with negative values
    c = np.array([-10.0, -5.5, 0.0, 5.5, 10.0, 15.0], dtype=np.float64)
    r = np.array([15.0, -2.0], dtype=np.float64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 4: 1D complex64 arrays
    c = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    r = np.array([4.0 - 1.0j, 0.0 + 0.0j, -2.0 - 2.0j, 1.0 + 3.0j], dtype=np.complex64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 5: 1D complex128 arrays
    c = np.array([0.5 - 1.5j, 2.0 + 3.0j], dtype=np.complex128)
    r = np.array([2.0 + 3.0j, -1.0 + 0.0j, 3.5 - 4.5j], dtype=np.complex128)
    list_of_inputs.append({"c": c, "r": r})

    # Input 6: 1D single-element arrays (minimal size)
    c = np.array([42.0], dtype=np.float32)
    r = np.array([42.0], dtype=np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 7: 2D batched float32 arrays with identical leading dimensions
    c = np.random.randn(3, 4).astype(np.float32)
    r = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 8: 2D broadcast-compatible leading dimensions
    c = np.random.randn(1, 6).astype(np.float64)
    r = np.random.randn(4, 3).astype(np.float64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 9: 3D batched int64 arrays
    c = np.random.randint(-100, 100, size=(2, 3, 5), dtype=np.int64)
    r = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 10: 4D multi-dimensional broadcasted arrays
    c = np.random.randn(2, 1, 3, 4).astype(np.float32)
    r = np.random.randn(1, 5, 3, 2).astype(np.float32)
    list_of_inputs.append({"c": c, "r": r})

    return list_of_inputs


generated_inputs["jax.scipy.linalg.hankel"] = hankel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.hankel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.hankel'.")


check_valid('jax.scipy.linalg.hankel', generated_inputs['jax.scipy.linalg.hankel'], lib="jax", suffix=0)
