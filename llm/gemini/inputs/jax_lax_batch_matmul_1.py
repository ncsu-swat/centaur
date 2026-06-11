
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def batch_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 3D tensor, "default" precision
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "default"
    })

    # Input 2: float32 3D tensor with single batch dimension, "high" precision
    lhs = np.random.randn(1, 5, 2).astype(np.float32)
    rhs = np.random.randn(1, 2, 3).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "high"
    })

    # Input 3: float64 3D tensor, "highest" precision
    lhs = np.random.randn(4, 8, 8).astype(np.float64)
    rhs = np.random.randn(4, 8, 8).astype(np.float64)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "highest"
    })

    # Input 4: 4D tensor, "default" precision
    lhs = np.random.randn(2, 2, 16, 8).astype(np.float32)
    rhs = np.random.randn(2, 2, 8, 32).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "default"
    })

    # Input 5: float32 with negative values, "high" precision
    lhs = np.random.uniform(-5.0, 5.0, (3, 4, 5)).astype(np.float32)
    rhs = np.random.uniform(-5.0, 5.0, (3, 5, 2)).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "high"
    })

    # Input 6: Large dimensions, "default" precision
    lhs = np.random.randn(8, 64, 128).astype(np.float32)
    rhs = np.random.randn(8, 128, 64).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "default"
    })

    # Input 7: Smallest possible dimensions 3D, "highest" precision
    lhs = np.random.randn(1, 1, 1).astype(np.float32)
    rhs = np.random.randn(1, 1, 1).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "highest"
    })

    # Input 8: High-dimensional batch tensor (5D), "default" precision
    lhs = np.random.randn(2, 1, 3, 4, 5).astype(np.float32)
    rhs = np.random.randn(2, 1, 3, 5, 6).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "default"
    })

    # Input 9: float64 with large values, "high" precision
    lhs = np.random.uniform(-100.0, 100.0, (2, 10, 10)).astype(np.float64)
    rhs = np.random.uniform(-100.0, 100.0, (2, 10, 10)).astype(np.float64)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "high"
    })

    # Input 10: float32 with asymmetric matrix shapes, "highest" precision
    lhs = np.random.randn(5, 12, 34).astype(np.float32)
    rhs = np.random.randn(5, 34, 15).astype(np.float32)
    list_of_inputs.append({
        "lhs": lhs,
        "rhs": rhs,
        "precision": "highest"
    })

    return list_of_inputs

generated_inputs["jax.lax.batch_matmul_1"] = batch_matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.batch_matmul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.batch_matmul_1'.")


check_valid('jax.lax.batch_matmul', generated_inputs['jax.lax.batch_matmul_1'], lib="jax", suffix=1)
