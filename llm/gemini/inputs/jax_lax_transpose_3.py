
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 transpose
    operand = np.random.randn(3, 5).astype(np.float32)
    permutation = np.array([1, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 2: 2D identity permutation float64
    operand = np.random.randn(4, 4).astype(np.float64)
    permutation = np.array([0, 1], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 3: 1D array
    operand = np.array([-2.5, 0.0, 1.5, -4.2], dtype=np.float32)
    permutation = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 4: 3D int32 array with cyclic permutation
    operand = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    permutation = np.array([1, 2, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 5: 3D reversed permutation
    operand = np.random.randn(2, 4, 6).astype(np.float32)
    permutation = np.array([2, 1, 0], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 6: 4D complex64 array
    operand = (
        np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)
    ).astype(np.complex64)
    permutation = np.array([0, 3, 1, 2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 7: 4D bool array
    operand = np.random.choice([True, False], size=(3, 2, 5, 4))
    permutation = np.array([3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 8: 5D float16 array
    operand = np.random.randn(2, 1, 3, 2, 4).astype(np.float16)
    permutation = np.array([4, 0, 3, 1, 2], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 9: Array with zero-sized dimension
    operand = np.empty((0, 3, 2), dtype=np.float32)
    permutation = np.array([2, 0, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 10: 6D int64 array
    operand = np.random.randint(-100, 100, size=(1, 2, 1, 3, 2, 1)).astype(
        np.int64
    )
    permutation = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    return list_of_inputs


generated_inputs["jax.lax.transpose_3"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.transpose_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.transpose_3'.")


check_valid('jax.lax.transpose', generated_inputs['jax.lax.transpose_3'], lib="jax", suffix=3)
