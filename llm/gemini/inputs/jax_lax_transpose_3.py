
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, simple transpose
    operand = np.random.randn(3, 4).astype(np.float32)
    permutation = np.array([1, 0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 2: 3D float32, permute axes (0, 2, 1)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    permutation = np.array([0, 2, 1], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 3: 4D float32, reverse all dimensions
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    permutation = np.array([3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 4: 1D float32, trivial transpose
    operand = np.random.randn(10).astype(np.float32)
    permutation = np.array([0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 5: 5D float32, complex permutation
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    permutation = np.array([0, 2, 4, 1, 3], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 6: 2D int32, simple transpose, permutation as int64 tensor
    operand = np.arange(12).reshape(3, 4).astype(np.int32)
    permutation = np.array([1, 0], dtype=np.int64)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 7: 3D float64, rotate axes (right shift)
    operand = np.random.randn(3, 4, 5).astype(np.float64)
    permutation = np.array([2, 0, 1], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 8: 3D bool array
    operand = (np.random.randn(2, 3, 2) > 0).astype(np.bool_)
    permutation = np.array([1, 2, 0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 9: 6D float32, reverse dimensions
    operand = np.random.randn(1, 2, 1, 3, 1, 4).astype(np.float32)
    permutation = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 10: 2D int64, simple transpose
    operand = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    permutation = np.array([1, 0], dtype=np.int32)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    return list_of_inputs

generated_inputs["jax.lax.transpose_3"] = jax_lax_transpose_inputs()

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
