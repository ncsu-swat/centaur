
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D array, concatenating along dimension 0 (float32)
    operands = np.random.randn(2, 3).astype(np.float32)
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, concatenating along dimension 0 (float32)
    operands = np.random.randn(3, 4, 5).astype(np.float32)
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, concatenating along dimension 1 with negative integer values (int32)
    operands = np.random.randint(-10, 10, size=(2, 4, 5), dtype=np.int32)
    dimension = 1
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, concatenating along dimension 2 (float64)
    operands = np.random.randn(5, 2, 3, 4).astype(np.float64)
    dimension = 2
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, concatenating along dimension 0 with negative values (int64)
    operands = np.random.randint(-100, 100, size=(4, 6, 8), dtype=np.int64)
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array, concatenating along dimension 0 (bool)
    operands = np.random.choice([True, False], size=(2, 5))
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D array, concatenating along dimension 3 (float32)
    operands = np.random.randn(3, 2, 2, 2, 2).astype(np.float32)
    dimension = 3
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single operand slice along axis 0 (complex64)
    operands = (np.random.randn(1, 4, 4) + 1j * np.random.randn(1, 4, 4)).astype(np.complex64)
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, concatenating along dimension 1 (uint8)
    operands = np.random.randint(0, 255, size=(6, 3, 2), dtype=np.uint8)
    dimension = 1
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, concatenating along dimension 1 (float16)
    operands = np.random.randn(2, 3, 5, 7).astype(np.float16)
    dimension = 1
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array, concatenating along dimension 0 (int16)
    operands = np.random.randint(-50, 50, size=(4, 2, 3), dtype=np.int16)
    dimension = 0
    input_dict = {"operands": operands, "dimension": dimension}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.concatenate_2"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.concatenate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.concatenate_2'.")


check_valid('jax.lax.concatenate', generated_inputs['jax.lax.concatenate_2'], lib="jax", suffix=2)
