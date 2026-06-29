
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def multi_dot_inputs():
    list_of_inputs = []

    # Input 1: 3 matrices of 10x10, float32, default precision
    list_of_inputs.append({
        "arrays": np.random.randn(3, 10, 10).astype(np.float32),
        "precision": "default"
    })

    # Input 2: 4 matrices of 5x5, float64, high precision
    list_of_inputs.append({
        "arrays": np.random.randn(4, 5, 5).astype(np.float64),
        "precision": "high"
    })

    # Input 3: 5 matrices of 4x4, float32, highest precision
    list_of_inputs.append({
        "arrays": np.random.randn(5, 4, 4).astype(np.float32),
        "precision": "highest"
    })

    # Input 4: 3 matrices of 6x6, scaled float32, default precision
    list_of_inputs.append({
        "arrays": (np.random.randn(3, 6, 6) * 10.0).astype(np.float32),
        "precision": "default"
    })

    # Input 5: 3 matrices of 8x8, float32, high precision
    list_of_inputs.append({
        "arrays": np.random.randn(3, 8, 8).astype(np.float32),
        "precision": "high"
    })

    # Input 6: 2 matrices of 12x12, float64, default precision
    list_of_inputs.append({
        "arrays": np.random.randn(2, 12, 12).astype(np.float64),
        "precision": "default"
    })

    # Input 7: 4 matrices of 3x3, float16, highest precision
    list_of_inputs.append({
        "arrays": np.random.randn(4, 3, 3).astype(np.float16),
        "precision": "highest"
    })

    # Input 8: 3 matrices of 7x7 with negative shift, float32, default precision
    list_of_inputs.append({
        "arrays": (np.random.randn(3, 7, 7) - 5.0).astype(np.float32),
        "precision": "default"
    })

    # Input 9: 6 matrices of 2x2, float32, high precision
    list_of_inputs.append({
        "arrays": np.random.randn(6, 2, 2).astype(np.float32),
        "precision": "high"
    })

    # Input 10: 3 matrices of 15x15, float64, default precision
    list_of_inputs.append({
        "arrays": np.random.randn(3, 15, 15).astype(np.float64),
        "precision": "default"
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.multi_dot_1"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.multi_dot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.multi_dot_1'.")


check_valid('jax.numpy.linalg.multi_dot', generated_inputs['jax.numpy.linalg.multi_dot_1'], lib="jax", suffix=1)
