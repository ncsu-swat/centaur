
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vecmat_inputs():
    list_of_inputs = []

    # Input 1: 1D vector and 2D matrix, float32
    x1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x2 = np.array(
        [[4.0, 5.0], [-6.0, 7.0], [8.0, -9.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D vector and 2D matrix, int32
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Batched 2D vector and 3D matrix, float64
    x1 = np.random.randn(5, 4).astype(np.float64)
    x2 = np.random.randn(5, 4, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting batch dimensions (1D vector with 3D matrix)
    x1 = np.random.randn(4).astype(np.float32)
    x2 = np.random.randn(3, 4, 6).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Multi-dimensional batch broadcasting
    x1 = np.random.randn(2, 1, 5).astype(np.float32)
    x2 = np.random.randn(1, 3, 5, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Complex numbers to test conjugate behavior
    x1 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    x2 = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Complex128 batched inputs
    x1 = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex128
    )
    x2 = (np.random.randn(4, 3, 5) + 1j * np.random.randn(4, 3, 5)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Integer int64 with negative values
    x1 = np.random.randint(-10, 10, size=(2, 4), dtype=np.int64)
    x2 = np.random.randint(-10, 10, size=(2, 4, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Single element inner dimension (M=1, N=1)
    x1 = np.array([[3.5]], dtype=np.float32)
    x2 = np.array([[[2.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Higher-rank tensors (4D and 5D with broadcast)
    x1 = np.random.randn(2, 1, 3, 6).astype(np.float32)
    x2 = np.random.randn(2, 4, 3, 6, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Boolean inputs
    x1 = np.array([True, False, True], dtype=bool)
    x2 = np.array(
        [[True, False], [False, True], [True, True]],
        dtype=bool,
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.vecmat"] = vecmat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vecmat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vecmat'.")


check_valid('jax.numpy.vecmat', generated_inputs['jax.numpy.vecmat'], lib="jax", suffix=0)
