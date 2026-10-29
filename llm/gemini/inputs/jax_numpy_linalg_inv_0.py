
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def inv_inputs():
    list_of_inputs = []

    # Input 1: 1x1 float32 matrix
    a = np.array([[5.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 2x2 float32 matrix with negative numbers
    a = np.array([[2.0, -1.0], [-3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 3x3 float64 matrix
    a = np.array(
        [[3.0, 1.0, 0.5], [1.0, 4.0, 2.0], [0.5, 2.0, 5.0]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 3D batched float32 matrices (batch size 4, 2x2)
    rng = np.random.RandomState(42)
    a = rng.randn(4, 2, 2).astype(np.float32)
    a += np.eye(2, dtype=np.float32) * 5.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: 4D batched float64 matrices (batch shape 2x3, 3x3)
    a = rng.randn(2, 3, 3, 3).astype(np.float64)
    a += np.eye(3, dtype=np.float64) * 6.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: 2x2 complex64 matrix
    a = (
        rng.randn(2, 2).astype(np.float32)
        + 1j * rng.randn(2, 2).astype(np.float32)
    )
    a += np.eye(2, dtype=np.complex64) * 5.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: 3x3 complex128 matrix
    a = (
        rng.randn(3, 3).astype(np.float64)
        + 1j * rng.randn(3, 3).astype(np.float64)
    )
    a += np.eye(3, dtype=np.complex128) * 6.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: Larger 10x10 float32 matrix
    a = rng.randn(10, 10).astype(np.float32)
    a += np.eye(10, dtype=np.float32) * 15.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 5D batched float32 matrices (1, 2, 1, 4, 4)
    a = rng.randn(1, 2, 1, 4, 4).astype(np.float32)
    a += np.eye(4, dtype=np.float32) * 8.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 4x4 diagonal float32 matrix with negative and positive values
    a = np.diag([-2.0, 3.5, -1.2, 4.0]).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: 3D batched complex64 matrices (3, 2, 2)
    a = (
        rng.randn(3, 2, 2).astype(np.float32)
        + 1j * rng.randn(3, 2, 2).astype(np.float32)
    )
    a += np.eye(2, dtype=np.complex64) * 5.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.inv"] = inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.inv'.")


check_valid('jax.numpy.linalg.inv', generated_inputs['jax.numpy.linalg.inv'], lib="jax", suffix=0)
