
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors, default precision
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([4.0, 2.0, -1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 2: 2D arrays with high precision
    a = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    b = np.array([[0.5, -1.5], [2.0, 2.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 3: Different leading dimensions
    a = np.random.randn(3, 4).astype(np.float64)
    b = np.random.randn(5, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 4: 3D and 2D arrays
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 5: Complex inputs
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    b = np.array([2.0 - 1.0j, 1.0 + 1.0j, -0.5 + 2.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": np.dtype(np.complex64),
            }
        )
    )

    # Input 6: Integer inputs with float preferred_element_type
    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    b = np.array([[7, 8, 9]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 7: 4D tensor with 1D vector
    a = np.random.randn(2, 2, 3, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 8: 1D vector with 3D tensor
    a = np.random.randn(8).astype(np.float32)
    b = np.random.randn(3, 2, 8).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 9: 3D and 3D tensors
    a = np.random.randn(2, 3, 4).astype(np.float64)
    b = np.random.randn(3, 2, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 10: Complex128 inputs
    a = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex128
    )
    b = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": np.dtype(np.complex128),
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.inner_1"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.inner_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.inner_1'.")


check_valid('jax.numpy.inner', generated_inputs['jax.numpy.inner_1'], lib="jax", suffix=1)
