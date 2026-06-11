
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D array, standard transpose
    operand = np.random.randn(3, 5).astype(np.float32)
    permutation = (1, 0)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 2: 3D array, permute last two dimensions
    operand = np.random.randn(2, 4, 6).astype(np.float32)
    permutation = (0, 2, 1)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 3: 3D array, shift all dimensions
    operand = np.random.randn(3, 4, 5).astype(np.float64)
    permutation = (2, 0, 1)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 4: 4D array, reverse all dimensions
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    permutation = (3, 2, 1, 0)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 5: 1D array, only one permutation possible
    operand = np.random.randn(10).astype(np.float32)
    permutation = (0,)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 6: 2D array with int32 type
    operand = np.random.randint(-10, 10, size=(4, 2)).astype(np.int32)
    permutation = (1, 0)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 7: 5D array with float16 type
    operand = np.random.randn(2, 2, 3, 3, 4).astype(np.float16)
    permutation = (0, 2, 4, 1, 3)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 8: 3D array with complex numbers
    operand = (np.random.randn(2, 3, 2) + 1j * np.random.randn(2, 3, 2)).astype(np.complex64)
    permutation = (1, 2, 0)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 9: 4D array with boolean values
    operand = np.random.choice([True, False], size=(2, 2, 2, 2))
    permutation = (2, 0, 3, 1)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    # Input 10: 3D array with int16 type and large dimensions
    operand = np.random.randint(-100, 100, size=(10, 15, 20)).astype(np.int16)
    permutation = (2, 1, 0)
    list_of_inputs.append({"operand": operand, "permutation": permutation})

    return list_of_inputs

generated_inputs["jax.lax.transpose_2"] = generate_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.transpose_2'.")


check_valid('jax.lax.transpose', generated_inputs['jax.lax.transpose_2'], lib="jax", suffix=2)
