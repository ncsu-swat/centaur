
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=dtype)


def batch_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D batch matmul with float32 and default precision
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 2: Higher precision (float64) with highest precision
    lhs = np.random.randn(4, 8, 16).astype(np.float64)
    rhs = np.random.randn(4, 16, 8).astype(np.float64)
    precision = PrecisionTuple(("highest", "highest"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 3: Float16 with high precision
    lhs = np.random.randn(1, 10, 10).astype(np.float16)
    rhs = np.random.randn(1, 10, 10).astype(np.float16)
    precision = PrecisionTuple(("high", "high"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 4: Negative and zero values
    lhs = np.array([[[-1.0, 0.0], [2.0, -3.0]]], dtype=np.float32)
    rhs = np.array([[[0.0, -5.0], [1.0, 4.0]]], dtype=np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 5: Single-element inner dimensions (vector outer product style in batch)
    lhs = np.random.randn(5, 7, 1).astype(np.float32)
    rhs = np.random.randn(5, 1, 6).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 6: Single-element outer dimensions (inner product style in batch)
    lhs = np.random.randn(3, 1, 12).astype(np.float32)
    rhs = np.random.randn(3, 12, 1).astype(np.float32)
    precision = PrecisionTuple(("high", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 7: Complex64 numbers
    lhs = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    rhs = (np.random.randn(2, 3, 5) + 1j * np.random.randn(2, 3, 5)).astype(
        np.complex64
    )
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 8: Complex128 numbers with asymmetric precision
    lhs = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex128
    )
    rhs = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex128
    )
    precision = PrecisionTuple(("highest", "high"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 9: Large batch size
    lhs = np.random.randn(64, 4, 8).astype(np.float32)
    rhs = np.random.randn(64, 8, 4).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

    # Input 10: Square matrices of odd dimensions
    lhs = np.random.randn(3, 7, 7).astype(np.float32)
    rhs = np.random.randn(3, 7, 7).astype(np.float32)
    precision = PrecisionTuple(("default", "highest"))
    list_of_inputs.append(
        copy.deepcopy({"lhs": lhs, "rhs": rhs, "precision": precision})
    )

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
