
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, dimension 0
    operands = np.random.randn(2, 5).astype(np.float32)
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 2: 3D float32 array, dimension 0
    operands = np.random.randn(3, 4, 5).astype(np.float32)
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 3: 3D float32 array, dimension 1
    operands = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append({"operands": operands, "dimension": 1})

    # Input 4: 4D int32 array, dimension 2
    operands = np.random.randint(-10, 10, size=(4, 2, 3, 2)).astype(np.int32)
    list_of_inputs.append({"operands": operands, "dimension": 2})

    # Input 5: 2D int32 array, dimension 0
    operands = np.random.randint(-10, 10, size=(5, 2)).astype(np.int32)
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 6: 3D float64 array, dimension 1
    operands = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"operands": operands, "dimension": 1})

    # Input 7: 3D boolean array, dimension 0
    operands = np.random.choice([True, False], size=(3, 2, 2)).astype(bool)
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 8: 3D int8 array, dimension 1
    operands = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int8)
    list_of_inputs.append({"operands": operands, "dimension": 1})

    # Input 9: 5D float32 array, dimension 3
    operands = np.random.randn(2, 1, 2, 1, 3).astype(np.float32)
    list_of_inputs.append({"operands": operands, "dimension": 3})

    # Input 10: 4D uint8 array, dimension 1
    operands = np.random.randint(0, 255, size=(3, 2, 2, 2)).astype(np.uint8)
    list_of_inputs.append({"operands": operands, "dimension": 1})

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
