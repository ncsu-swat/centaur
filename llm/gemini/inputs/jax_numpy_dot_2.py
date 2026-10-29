
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class PrecisionTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array([0, 0], dtype=np.int32)


def dot_inputs():
    list_of_inputs = []

    # Input 1: 1D vectors with default precision and float32 dtype
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([-4.0, 5.0, 6.0], dtype=np.float32)
    precision = PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 2: 2D matrices with high precision and float64 dtype
    a = np.random.randn(3, 4).astype(np.float64)
    b = np.random.randn(4, 5).astype(np.float64)
    precision = PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGH))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 3: 2D matrix and 1D vector with highest precision
    a = np.random.randn(5, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 4: 1D vector and 2D matrix
    a = np.random.randn(4).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 5: 3D tensor and 2D matrix
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 6: 3D tensor and 1D vector
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 7: 3D tensor and 3D tensor
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(5, 4, 6).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGH))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 8: Integer arrays with int32 accumulation
    a = np.random.randint(-10, 10, size=(4, 5), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(5, 2), dtype=np.int32)
    precision = PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT))
    preferred_element_type = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 9: 0D scalar arrays
    a = np.array(3.0, dtype=np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 10: 4D tensor with 2D matrix
    a = np.random.randn(2, 1, 3, 5).astype(np.float32)
    b = np.random.randn(5, 4).astype(np.float32)
    precision = PrecisionTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.dot_2"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dot_2'.")


check_valid('jax.numpy.dot', generated_inputs['jax.numpy.dot_2'], lib="jax", suffix=2)
