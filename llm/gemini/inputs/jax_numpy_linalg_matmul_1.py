
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_matmul_inputs():
    list_of_inputs = []

    # Input 1: 1D vector dot product (float32)
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'default',
        'preferred_element_type': np.float32
    })

    # Input 2: 2D matrix multiplication (float32)
    x1 = np.random.randn(4, 3).astype(np.float32)
    x2 = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'high',
        'preferred_element_type': np.float32
    })

    # Input 3: Matrix-vector multiplication (float64, negative values)
    x1 = np.random.uniform(-10, 10, size=(3, 4)).astype(np.float64)
    x2 = np.random.uniform(-10, 10, size=(4,)).astype(np.float64)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'highest',
        'preferred_element_type': np.float64
    })

    # Input 4: Batched matrix multiplication (3D x 3D)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'default',
        'preferred_element_type': np.float32
    })

    # Input 5: Broadcasted batch multiplication (3D x 2D)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'high',
        'preferred_element_type': np.float32
    })

    # Input 6: Integer matrix multiplication
    x1 = np.random.randint(-5, 5, size=(3, 2)).astype(np.int32)
    x2 = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'default',
        'preferred_element_type': np.int32
    })

    # Input 7: Large matrix multiplication
    x1 = np.random.randn(64, 128).astype(np.float32)
    x2 = np.random.randn(128, 64).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'highest',
        'preferred_element_type': np.float32
    })

    # Input 8: 4D tensor multiplication
    x1 = np.random.randn(2, 2, 4, 3).astype(np.float32)
    x2 = np.random.randn(2, 2, 3, 5).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'default',
        'preferred_element_type': np.float32
    })

    # Input 9: Mixed precision (float16 inputs, float32 accumulation)
    x1 = np.random.randn(4, 4).astype(np.float16)
    x2 = np.random.randn(4, 4).astype(np.float16)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'high',
        'preferred_element_type': np.float32
    })

    # Input 10: Vector-Matrix multiplication (1D x 2D)
    x1 = np.random.randn(3).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': 'highest',
        'preferred_element_type': np.float32
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matmul_1"] = generate_matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matmul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matmul_1'.")


check_valid('jax.numpy.linalg.matmul', generated_inputs['jax.numpy.linalg.matmul_1'], lib="jax", suffix=1)
