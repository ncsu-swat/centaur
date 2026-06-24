
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix with 1D vector, upper triangular
    a = np.random.randn(3, 3).astype(np.float32) + np.eye(3).astype(np.float32) * 3.0
    b = np.random.randn(3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular, 2D b
    a = np.random.randn(5, 5).astype(np.float32) + np.eye(5).astype(np.float32) * 4.0
    b = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transposed solver (trans=1)
    a = np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 3.0
    b = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Conjugate transposed solver (trans=2)
    a = np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 3.0
    b = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 2,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Unit diagonal = True
    a = np.random.randn(6, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": True,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched inputs a: (2, 4, 4), b: (2, 4)
    a = np.random.randn(2, 4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 3.0
    b = np.random.randn(2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 dtype
    a = np.random.randn(5, 5).astype(np.float64) + np.eye(5).astype(np.float64) * 5.0
    b = np.random.randn(5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched inputs with 2D b: a: (3, 5, 5), b: (3, 5, 2)
    eye_batch = np.stack([np.eye(5)] * 3)
    a = np.random.randn(3, 5, 5).astype(np.float32) + eye_batch.astype(np.float32) * 4.0
    b = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": True,
        "debug": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions, single precision
    a = np.random.randn(10, 10).astype(np.float32) + np.eye(10).astype(np.float32) * 10.0
    b = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": True,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive values with larger batch size (5, 3, 3)
    eye_batch_10 = np.stack([np.eye(3)] * 5)
    a = np.random.randn(5, 3, 3).astype(np.float32) * 10.0 + eye_batch_10.astype(np.float32) * 20.0
    b = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.solve_triangular_1"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_triangular_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_triangular_1'.")


check_valid('jax.scipy.linalg.solve_triangular', generated_inputs['jax.scipy.linalg.solve_triangular_1'], lib="jax", suffix=1)
