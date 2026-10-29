
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array(list(self), dtype=object)


def matmul_inputs():
    list_of_inputs = []

    # Input 1: 1D @ 1D vector dot product, float32
    x1 = np.array([1.5, -2.0, 3.2], dtype=np.float32)
    x2 = np.array([-0.5, 4.0, 1.0], dtype=np.float32)
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 2: 2D @ 2D matrix multiplication, float32 with high precision
    x1 = np.random.randn(4, 5).astype(np.float32)
    x2 = np.random.randn(5, 3).astype(np.float32)
    precision = PrecisionTuple(("high", "high"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 3: 2D @ 1D matrix-vector multiplication, float64
    x1 = np.random.randn(6, 4).astype(np.float64)
    x2 = np.random.randn(4).astype(np.float64)
    precision = PrecisionTuple(("highest", "highest"))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 4: 1D @ 2D vector-matrix multiplication, float32
    x1 = np.random.randn(3).astype(np.float32)
    x2 = np.random.randn(3, 7).astype(np.float32)
    precision = PrecisionTuple(("default", "high"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 5: Batched 3D @ 3D multiplication, float32
    x1 = np.random.randn(8, 3, 4).astype(np.float32)
    x2 = np.random.randn(8, 4, 2).astype(np.float32)
    precision = PrecisionTuple(("highest", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 6: Broadcasted batch dimensions (1, 4, 3) @ (5, 3, 2), float32
    x1 = np.random.randn(1, 4, 3).astype(np.float32)
    x2 = np.random.randn(5, 3, 2).astype(np.float32)
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 7: 4D tensors with broadcasting, float32
    x1 = np.random.randn(2, 1, 5, 6).astype(np.float32)
    x2 = np.random.randn(1, 3, 6, 4).astype(np.float32)
    precision = PrecisionTuple(("high", "high"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 8: Complex64 matrices, 2D @ 2D
    x1 = (
        np.random.randn(3, 3).astype(np.float32)
        + 1j * np.random.randn(3, 3).astype(np.float32)
    ).astype(np.complex64)
    x2 = (
        np.random.randn(3, 3).astype(np.float32)
        + 1j * np.random.randn(3, 3).astype(np.float32)
    ).astype(np.complex64)
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 9: Integer arrays, 2D @ 2D, accumulated to int32
    x1 = np.random.randint(-10, 10, size=(4, 4), dtype=np.int32)
    x2 = np.random.randint(-10, 10, size=(4, 4), dtype=np.int32)
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 10: 3D batched @ 1D vector (batched matrix-vector), float64
    x1 = np.random.randn(4, 5, 6).astype(np.float64)
    x2 = np.random.randn(6).astype(np.float64)
    precision = PrecisionTuple(("highest", "highest"))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matmul_2"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matmul_2'.")


check_valid('jax.numpy.linalg.matmul', generated_inputs['jax.numpy.linalg.matmul_2'], lib="jax", suffix=2)
