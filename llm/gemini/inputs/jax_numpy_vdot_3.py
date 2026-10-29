
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors, default precision, float32 accumulation
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([-0.5, 4.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": "float32",
            }
        )
    )

    # Input 2: 1D complex64 vectors, high precision, complex64 accumulation
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    b = np.array([2.0 - 1.0j, 0.0 + 3.0j, -1.0 - 2.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": "complex64",
            }
        )
    )

    # Input 3: 2D arrays with identical total size, highest precision, float64 accumulation
    a = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": "float64",
            }
        )
    )

    # Input 4: Different shapes (2D vs 1D) but same size
    a = np.arange(6, dtype=np.float32).reshape(2, 3) - 3.0
    b = np.arange(6, dtype=np.float32) + 1.0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": "float32",
            }
        )
    )

    # Input 5: 3D complex128 arrays
    a = np.ones((2, 2, 2), dtype=np.complex128) * (1.0 - 1.0j)
    b = np.full((2, 2, 2), 2.0 + 3.0j, dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": "complex128",
            }
        )
    )

    # Input 6: Integer inputs with float32 accumulation
    a = np.array([-5, 10, -15, 20], dtype=np.int32)
    b = np.array([2, -3, 4, -1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": "float32",
            }
        )
    )

    # Input 7: 4D reshaped arrays, highest precision
    a = np.random.randn(2, 3, 2, 2).astype(np.float32)
    b = np.random.randn(24).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": "float32",
            }
        )
    )

    # Input 8: Boolean arrays
    a = np.array([True, False, True, True], dtype=np.bool_)
    b = np.array([False, True, True, False], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": "int32",
            }
        )
    )

    # Input 9: Single-element 0D/1D arrays
    a = np.array([3.14], dtype=np.float64)
    b = np.array([2.71], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "high",
                "preferred_element_type": "float64",
            }
        )
    )

    # Input 10: Mixed shapes with float16 input, float32 accumulation
    a = np.array([[-1.5, 2.0], [0.5, -3.0]], dtype=np.float16)
    b = np.array([[4.0], [-1.0], [2.0], [0.0]], dtype=np.float16)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "default",
                "preferred_element_type": "float32",
            }
        )
    )

    # Input 11: Complex vectors with mixed real and imaginary parts
    a = np.array([-2.5 + 1.5j, 0.0 - 3.0j], dtype=np.complex64)
    b = np.array([1.0 - 0.5j, 4.0 + 2.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "precision": "highest",
                "preferred_element_type": "complex64",
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.vdot_3"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_3'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_3'], lib="jax", suffix=3)
