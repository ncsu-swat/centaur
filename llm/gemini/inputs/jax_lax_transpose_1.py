
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_transpose_inputs():
    list_of_inputs = []

    # Input 1: 1D array, identity permutation
    operand = np.array([1.0, -2.5, 3.0, -4.5], dtype=np.float32)
    permutation = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 2: 2D standard matrix transpose
    operand = np.arange(12, dtype=np.int32).reshape(3, 4) - 5
    permutation = [1, 0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 3: 2D identity permutation
    operand = np.random.randn(5, 5).astype(np.float64)
    permutation = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 4: 3D array cyclic shift
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    permutation = [1, 2, 0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 5: 3D array reverse axes
    operand = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    permutation = [2, 1, 0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 6: 4D batch tensor NCHW to NHWC
    operand = np.random.randn(8, 3, 32, 32).astype(np.float32)
    permutation = [0, 2, 3, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 7: 4D tensor with complex numbers
    operand = (
        np.random.randn(2, 4, 6, 8) + 1j * np.random.randn(2, 4, 6, 8)
    ).astype(np.complex64)
    permutation = [3, 1, 0, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 8: 5D high-dimensional tensor with unit dimensions
    operand = np.ones((1, 2, 1, 4, 3), dtype=np.int8)
    permutation = [4, 3, 2, 1, 0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 9: 0D scalar tensor
    operand = np.array(-42.0, dtype=np.float32)
    permutation = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 10: 3D unsigned integer array
    operand = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    permutation = [0, 2, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    return list_of_inputs


generated_inputs["jax.lax.transpose_1"] = lax_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.transpose_1'.")


check_valid('jax.lax.transpose', generated_inputs['jax.lax.transpose_1'], lib="jax", suffix=1)
