
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cho_factor_inputs():
    list_of_inputs = []

    # Helper function to generate positive-definite matrices
    def make_spd(shape, dtype=np.float32):
        n = shape[-1]
        if np.issubdtype(dtype, np.complexfloating):
            m = np.random.randn(*shape).astype(
                dtype
            ) + 1j * np.random.randn(*shape).astype(dtype)
            m_h = np.swapaxes(m.conj(), -1, -2)
            eye = np.eye(n, dtype=dtype)
            return np.matmul(m, m_h) + eye * (n + 1.0)
        else:
            m = np.random.randn(*shape).astype(dtype)
            m_t = np.swapaxes(m, -1, -2)
            eye = np.eye(n, dtype=dtype)
            return np.matmul(m, m_t) + eye * (n + 1.0)

    # Input 1: 2x2 float32 matrix, default flags
    a = make_spd((2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": False,
                "overwrite_a": False,
                "check_finite": True,
            }
        )
    )

    # Input 2: 3x3 float64 matrix, lower=True
    a = make_spd((3, 3), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": False,
                "check_finite": False,
            }
        )
    )

    # Input 3: Batched 3D float32 matrices (4, 5, 5)
    a = make_spd((4, 5, 5), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": True,
                "check_finite": True,
            }
        )
    )

    # Input 4: 3x3 complex64 Hermitian positive-definite matrix
    a = make_spd((3, 3), dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": False,
                "overwrite_a": False,
                "check_finite": True,
            }
        )
    )

    # Input 5: 4x4 complex128 Hermitian positive-definite matrix
    a = make_spd((4, 4), dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": True,
                "check_finite": False,
            }
        )
    )

    # Input 6: 1x1 float32 scalar-like matrix
    a = np.array([[4.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": False,
                "overwrite_a": False,
                "check_finite": False,
            }
        )
    )

    # Input 7: Batched 4D float64 matrices (2, 3, 4, 4)
    a = make_spd((2, 3, 4, 4), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": False,
                "overwrite_a": False,
                "check_finite": True,
            }
        )
    )

    # Input 8: Larger 10x10 float32 matrix
    a = make_spd((10, 10), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": False,
                "check_finite": True,
            }
        )
    )

    # Input 9: Batched complex64 matrices (2, 3, 3)
    a = make_spd((2, 3, 3), dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": False,
                "check_finite": False,
            }
        )
    )

    # Input 10: 6x6 float64 matrix with all flags True
    a = make_spd((6, 6), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "lower": True,
                "overwrite_a": True,
                "check_finite": True,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.cho_factor"] = cho_factor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cho_factor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cho_factor'.")


check_valid('jax.scipy.linalg.cho_factor', generated_inputs['jax.scipy.linalg.cho_factor'], lib="jax", suffix=0)
