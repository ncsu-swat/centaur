
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

jax.lax.Precision.__lt__ = lambda self, other: False
jax.lax.Precision.__le__ = lambda self, other: True
jax.lax.Precision.__gt__ = lambda self, other: False
jax.lax.Precision.__ge__ = lambda self, other: True

def batch_matmul_inputs():
    list_of_inputs = []

    # Input 1
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 2
    lhs = np.random.randn(1, 5, 8).astype(np.float32)
    rhs = np.random.randn(1, 8, 3).astype(np.float32)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 3
    lhs = np.random.randn(4, 10, 2).astype(np.float64)
    rhs = np.random.randn(4, 2, 6).astype(np.float64)
    precision = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 4
    lhs = np.random.randn(2, 2, 4, 3).astype(np.float32)
    rhs = np.random.randn(2, 2, 3, 5).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 5
    lhs = (np.random.randn(8, 16, 16) + 1j * np.random.randn(8, 16, 16)).astype(np.complex64)
    rhs = (np.random.randn(8, 16, 16) + 1j * np.random.randn(8, 16, 16)).astype(np.complex64)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 6
    lhs = np.random.randn(3, 7, 5).astype(np.float32)
    rhs = np.random.randn(3, 5, 2).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 7
    lhs = np.random.randn(5, 1, 3).astype(np.float32)
    rhs = np.random.randn(5, 3, 10).astype(np.float32)
    precision = (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 8
    lhs = np.random.randn(2, 3, 12, 8).astype(np.float64)
    rhs = np.random.randn(2, 3, 8, 15).astype(np.float64)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 9
    lhs = np.random.uniform(-10.0, 10.0, (10, 4, 4)).astype(np.float32)
    rhs = np.random.uniform(-10.0, 10.0, (10, 4, 4)).astype(np.float32)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    # Input 10
    lhs = np.random.randn(1, 1, 1, 2, 2).astype(np.float32)
    rhs = np.random.randn(1, 1, 1, 2, 2).astype(np.float32)
    precision = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    list_of_inputs.append({"lhs": lhs, "rhs": rhs, "precision": precision})

    return list_of_inputs

generated_inputs["jax.lax.batch_matmul_2"] = batch_matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.batch_matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.batch_matmul_2'.")


check_valid('jax.lax.batch_matmul', generated_inputs['jax.lax.batch_matmul_2'], lib="jax", suffix=2)
