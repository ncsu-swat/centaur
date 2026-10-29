
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=np.int32)


def matmul_inputs():
    list_of_inputs = []

    # Input 1: 1D @ 1D (vector dot product)
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([-0.5, 4.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D @ 2D matrix multiplication with negative values
    a = np.array([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=np.float32)
    b = np.array([[2.0, -1.0], [0.5, 3.0], [-2.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D @ 2D vector-matrix multiplication
    a = np.array([1.5, -2.5, 0.5, 3.0], dtype=np.float64)
    b = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
        ),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D @ 1D matrix-vector multiplication
    a = np.random.randn(6, 4).astype(np.float32)
    b = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched matrix multiplication
    a = np.random.randn(3, 4, 5).astype(np.float32)
    b = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasted leading dimensions (1, 4, 3) @ (2, 3, 5)
    a = np.random.randn(1, 4, 3).astype(np.float32)
    b = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched tensor multiplication
    a = np.random.randn(2, 3, 4, 6).astype(np.float32)
    b = np.random.randn(2, 3, 6, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex-valued matrix multiplication
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    b = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Accumulating float32 inputs into float64 output
    a = np.random.randn(5, 7).astype(np.float32)
    b = np.random.randn(7, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional broadcasting with 4D @ 3D
    a = np.random.randn(2, 1, 5, 3).astype(np.float32)
    b = np.random.randn(4, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.matmul_2"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matmul_2'.")


check_valid('jax.numpy.matmul', generated_inputs['jax.numpy.matmul_2'], lib="jax", suffix=2)
