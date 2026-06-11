
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reshape_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D to 2D reshape (float32)
    operand = np.arange(6, dtype=np.float32)
    new_sizes = (2, 3)
    dimensions = (0,)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 2: 2D to 1D reshape (int32)
    operand = np.arange(6, dtype=np.int32).reshape(2, 3)
    new_sizes = (6,)
    dimensions = (0, 1)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 3: 2D to 1D reshape with dimension permutation (float64)
    operand = np.arange(6, dtype=np.float64).reshape(2, 3)
    new_sizes = (6,)
    dimensions = (1, 0)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 4: 3D to 3D with dimension permutation (bool)
    operand = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    new_sizes = (2, 2, 2)
    dimensions = (2, 0, 1)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 5: 1D to 3D reshape (int16)
    operand = np.arange(24, dtype=np.int16)
    new_sizes = (2, 3, 4)
    dimensions = (0,)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 6: 3D to 2D reshape (float32)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = (6, 4)
    dimensions = (1, 0, 2)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 7: 4D to 1D reshape (uint8)
    operand = np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2)
    new_sizes = (16,)
    dimensions = (3, 2, 1, 0)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 8: 2D to 2D reshape (complex64)
    operand = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    new_sizes = (2, 8)
    dimensions = (0, 1)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 9: 1-element 3D to 1D reshape (float32)
    operand = np.array([[[1.5]]], dtype=np.float32)
    new_sizes = (1,)
    dimensions = (2, 1, 0)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    # Input 10: 5D to 2D reshape (int64)
    operand = np.arange(32, dtype=np.int64).reshape(2, 2, 2, 2, 2)
    new_sizes = (8, 4)
    dimensions = (4, 3, 2, 1, 0)
    list_of_inputs.append({
        "operand": copy.deepcopy(operand),
        "new_sizes": new_sizes,
        "dimensions": dimensions
    })

    return list_of_inputs

generated_inputs["jax.lax.reshape_1"] = generate_reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_1'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_1'], lib="jax", suffix=1)
