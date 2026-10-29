
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cholesky_inputs():
    list_of_inputs = []

    # Helper function to generate positive-definite matrix
    def make_spd(shape, dtype=np.float32):
        n = shape[-1]
        if np.issubdtype(dtype, np.complexfloating):
            real_part = np.random.randn(*shape)
            imag_part = np.random.randn(*shape)
            m = (real_part + 1j * imag_part).astype(dtype)
            m_conj = np.swapaxes(m.conj(), -1, -2)
            eye = np.eye(n, dtype=dtype)
            res = m @ m_conj + eye * n
        else:
            m = np.random.randn(*shape).astype(dtype)
            m_t = np.swapaxes(m, -1, -2)
            eye = np.eye(n, dtype=dtype)
            res = m @ m_t + eye * n
        return res

    # Input 1: 2x2 float32 matrix, upper decomposition
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

    # Input 2: 3x3 float64 matrix, lower decomposition
    a = make_spd((3, 3), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": False, "check_finite": False}
        )
    )

    # Input 3: 5x5 float32 matrix, upper decomposition with overwrite_a=True
    a = make_spd((5, 5), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": False, "overwrite_a": True, "check_finite": True}
        )
    )

    # Input 4: 1x1 matrix
    a = np.array([[4.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": False, "check_finite": True}
        )
    )

    # Input 5: Batched 3D array (4, 3, 3) float32
    a = make_spd((4, 3, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": False, "check_finite": True}
        )
    )

    # Input 6: Batched 4D array (2, 3, 4, 4) float64
    a = make_spd((2, 3, 4, 4), dtype=np.float64)
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

    # Input 7: Complex64 Hermitian positive-definite matrix (4, 4)
    a = make_spd((4, 4), dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": True, "check_finite": False}
        )
    )

    # Input 8: Complex128 Hermitian positive-definite matrix (3, 3)
    a = make_spd((3, 3), dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": False, "overwrite_a": False, "check_finite": True}
        )
    )

    # Input 9: Batched complex Hermitian positive-definite matrix (2, 3, 3)
    a = make_spd((2, 3, 3), dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": True, "check_finite": True}
        )
    )

    # Input 10: Larger 16x16 float32 matrix
    a = make_spd((16, 16), dtype=np.float32)
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

    # Input 11: 4x4 Identity matrix float64
    a = np.eye(4, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "lower": True, "overwrite_a": False, "check_finite": True}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cholesky'.")


check_valid('jax.scipy.linalg.cholesky', generated_inputs['jax.scipy.linalg.cholesky'], lib="jax", suffix=0)
