
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matvec_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix and 1D vector with float32
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: Integer types with negative values
    x1 = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    x2 = np.array([-2, 0, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: Batched matrix-vector multiplication (matching batch dim)
    x1 = np.random.randn(4, 3, 5).astype(np.float64)
    x2 = np.random.randn(4, 5).astype(np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting in batch dimensions (x1 has batch dim, x2 is unbatched)
    x1 = np.random.randn(5, 2, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Broadcasting in batch dimensions (x1 is unbatched, x2 has batch dim)
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(6, 4).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: Higher-dimensional batching with broadcast compatibility (e.g., (2, 1, 3, 4) and (1, 5, 4))
    x1 = np.random.randn(2, 1, 3, 4).astype(np.float32)
    x2 = np.random.randn(1, 5, 4).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: Complex numbers
    x1 = (
        np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    ).astype(np.complex64)
    x2 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: 1x1 matrix and 1-element vector
    x1 = np.array([[-5.5]], dtype=np.float32)
    x2 = np.array([2.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Boolean arrays
    x1 = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    x2 = np.array([True, True, False], dtype=np.bool_)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: 4D batching with matching shapes
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 11: Non-square large dimensions
    x1 = np.random.randn(10, 100, 50).astype(np.float32)
    x2 = np.random.randn(10, 50).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs


generated_inputs["jax.numpy.matvec"] = matvec_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matvec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matvec'.")


check_valid('jax.numpy.matvec', generated_inputs['jax.numpy.matvec'], lib="jax", suffix=0)
