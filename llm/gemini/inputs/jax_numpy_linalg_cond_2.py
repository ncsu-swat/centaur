
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cond_inputs():
    list_of_inputs = []

    # Input 1: 2x2 square float32 matrix, p = 2.0
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    p = 2.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 2: 3x3 square float64 matrix, p = 1.0
    x = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    p = 1.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 3: 4x4 square float32 matrix, p = -1.0
    x = np.random.randn(4, 4).astype(np.float32)
    p = -1.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 4: 2x3 non-square float32 matrix, p = 2.0
    x = np.random.randn(2, 3).astype(np.float32)
    p = 2.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 5: 5x5 square float64 matrix, p = inf
    x = np.random.randn(5, 5).astype(np.float64)
    p = float('inf')
    list_of_inputs.append({"x": x, "p": p})

    # Input 6: 3x3 square float32 matrix, p = -inf
    x = np.random.randn(3, 3).astype(np.float32)
    p = float('-inf')
    list_of_inputs.append({"x": x, "p": p})

    # Input 7: 4x2 non-square float64 matrix, p = -2.0
    x = np.random.randn(4, 2).astype(np.float64)
    p = -2.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 8: 10x10 square float32 matrix with negative values, p = 2.0
    x = np.random.uniform(-10.0, 10.0, (10, 10)).astype(np.float32)
    p = 2.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 9: Batched square float32 matrix (2, 3, 3), p = 1.0
    x = np.random.randn(2, 3, 3).astype(np.float32)
    p = 1.0
    list_of_inputs.append({"x": x, "p": p})

    # Input 10: Batched non-square float64 matrix (3, 4, 2), p = -2.0
    x = np.random.randn(3, 4, 2).astype(np.float64)
    p = -2.0
    list_of_inputs.append({"x": x, "p": p})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.cond_2"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cond_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cond_2'.")


check_valid('jax.numpy.linalg.cond', generated_inputs['jax.numpy.linalg.cond_2'], lib="jax", suffix=2)
