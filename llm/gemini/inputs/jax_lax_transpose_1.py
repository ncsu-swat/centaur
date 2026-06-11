
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, transpose axes
    operand = np.random.randn(3, 5).astype(np.float32)
    permutation = [1, 0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 2: 3D float32, shift axes right
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    permutation = [2, 0, 1]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 3: 1D int32, trivial permutation
    operand = np.arange(10).astype(np.int32)
    permutation = [0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 4: 4D int64, reverse axes
    operand = np.random.randint(0, 10, size=(2, 3, 4, 5)).astype(np.int64)
    permutation = [3, 2, 1, 0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 5: 3D float64, transpose last two axes
    operand = np.random.randn(4, 5, 6).astype(np.float64)
    permutation = [0, 2, 1]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 6: 5D float32, reverse axes
    operand = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    permutation = [4, 3, 2, 1, 0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 7: 2D int32, transpose axes
    operand = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    permutation = [1, 0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 8: 3D float64, shift axes left
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    permutation = [1, 2, 0]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 9: 4D float32, transpose inner axes
    operand = np.random.randn(2, 3, 3, 2).astype(np.float32)
    permutation = [0, 2, 1, 3]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 10: 5D int32, complex permutation
    operand = np.random.randint(0, 100, size=(2, 1, 3, 1, 4)).astype(np.int32)
    permutation = [4, 0, 2, 1, 3]
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    return list_of_inputs

generated_inputs["jax.lax.transpose_1"] = generate_transpose_inputs()

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
