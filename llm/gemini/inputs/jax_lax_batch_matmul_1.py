
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def batch_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 with "default" precision
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "default"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 with negative values and "high" precision
    lhs = np.random.uniform(-5.0, 5.0, size=(4, 5, 2)).astype(np.float64)
    rhs = np.random.uniform(-5.0, 5.0, size=(4, 2, 3)).astype(np.float64)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "high"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 with "highest" precision, single element matrices
    lhs = np.array([[[2.5]]], dtype=np.float32)
    rhs = np.array([[[-1.5]]], dtype=np.float32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "highest"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 with "default" precision
    lhs = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    rhs = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "default"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int32 inputs with "default" precision
    lhs = np.random.randint(-10, 10, size=(2, 4, 3), dtype=np.int32)
    rhs = np.random.randint(-10, 10, size=(2, 3, 2), dtype=np.int32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "default"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float16 inputs with "fastest" precision
    lhs = np.random.randn(5, 8, 4).astype(np.float16)
    rhs = np.random.randn(5, 4, 6).astype(np.float16)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "fastest"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large batch size with "high" precision
    lhs = np.random.randn(16, 2, 3).astype(np.float32)
    rhs = np.random.randn(16, 3, 2).astype(np.float32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "high"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero-sized dimension with "default" precision
    lhs = np.zeros((2, 0, 4), dtype=np.float32)
    rhs = np.zeros((2, 4, 3), dtype=np.float32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "default"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch size 1 with larger inner dimensions and "highest" precision
    lhs = np.random.randn(1, 32, 16).astype(np.float32)
    rhs = np.random.randn(1, 16, 8).astype(np.float32)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "highest"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex128 with "high" precision
    lhs = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    rhs = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "high"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Int64 inputs with "default" precision
    lhs = np.random.randint(-5, 5, size=(3, 2, 4), dtype=np.int64)
    rhs = np.random.randint(-5, 5, size=(3, 4, 1), dtype=np.int64)
    input_dict = {"lhs": lhs, "rhs": rhs, "precision": "default"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
