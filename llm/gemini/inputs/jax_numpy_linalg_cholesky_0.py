
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cholesky_inputs():
    list_of_inputs = []

    # Helper to generate positive definite matrices
    def make_spd(shape, dtype=np.float32):
        n = shape[-1]
        a = np.random.randn(*shape).astype(dtype)
        if np.iscomplexobj(a):
            a = a + 1j * np.random.randn(*shape).astype(dtype)
            conj_t = np.swapaxes(a.conj(), -1, -2)
            spd = np.matmul(a, conj_t)
            eye = np.eye(n, dtype=dtype)
            return spd + eye * (n * 2.0)
        else:
            t = np.swapaxes(a, -1, -2)
            spd = np.matmul(a, t)
            eye = np.eye(n, dtype=dtype)
            return spd + eye * (n * 2.0)

    np.random.seed(42)

    # Input 1: Basic 2D float32 SPD matrix, default booleans
    a1 = make_spd((3, 3), np.float32)
    list_of_inputs.append(
        {"a": a1, "upper": False, "symmetrize_input": True}
    )

    # Input 2: 2D float64 SPD matrix, upper=True
    a2 = make_spd((4, 4), np.float64)
    list_of_inputs.append(
        {"a": a2, "upper": True, "symmetrize_input": True}
    )

    # Input 3: 2D float32 with symmetrize_input=False
    a3 = make_spd((5, 5), np.float32)
    list_of_inputs.append(
        {"a": a3, "upper": False, "symmetrize_input": False}
    )

    # Input 4: 2D float64, upper=True and symmetrize_input=False
    a4 = make_spd((6, 6), np.float64)
    list_of_inputs.append(
        {"a": a4, "upper": True, "symmetrize_input": False}
    )

    # Input 5: Batched 3D float32 SPD matrices
    a5 = make_spd((2, 4, 4), np.float32)
    list_of_inputs.append(
        {"a": a5, "upper": False, "symmetrize_input": True}
    )

    # Input 6: Batched 3D float64 SPD matrices, upper=True
    a6 = make_spd((5, 3, 3), np.float64)
    list_of_inputs.append(
        {"a": a6, "upper": True, "symmetrize_input": False}
    )

    # Input 7: Higher-dimensional 4D batch float32
    a7 = make_spd((2, 3, 2, 2), np.float32)
    list_of_inputs.append(
        {"a": a7, "upper": False, "symmetrize_input": True}
    )

    # Input 8: 1x1 matrix edge case
    a8 = np.array([[4.0]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a8, "upper": True, "symmetrize_input": True}
    )

    # Input 9: Complex64 Hermitian positive definite matrix
    a9 = make_spd((3, 3), np.complex64)
    list_of_inputs.append(
        {"a": a9, "upper": False, "symmetrize_input": True}
    )

    # Input 10: Complex128 Hermitian positive definite matrix, upper=True
    a10 = make_spd((4, 4), np.complex128)
    list_of_inputs.append(
        {"a": a10, "upper": True, "symmetrize_input": False}
    )

    # Input 11: Batched complex128 3D matrix
    a11 = make_spd((3, 2, 2), np.complex128)
    list_of_inputs.append(
        {"a": a11, "upper": False, "symmetrize_input": False}
    )

    # Input 12: Matrix with explicit negative off-diagonal entries
    a12 = np.array(
        [[10.0, -2.0, -1.0], [-2.0, 10.0, -3.0], [-1.0, -3.0, 10.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(
        {"a": a12, "upper": True, "symmetrize_input": True}
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cholesky'.")


check_valid('jax.numpy.linalg.cholesky', generated_inputs['jax.numpy.linalg.cholesky'], lib="jax", suffix=0)
