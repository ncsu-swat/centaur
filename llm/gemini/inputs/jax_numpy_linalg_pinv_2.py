
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pinv_inputs():
    list_of_inputs = []

    # Input 1: 2D tall matrix (M > N), float32, hermitian=False, rtol specified
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rtol = np.array(1e-5, dtype=np.float32)
    hermitian = False
    rcond = None
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 2: 2D wide matrix (M < N), float64 with negative values, hermitian=False, rcond specified
    a = np.array([[-2.5, 1.5, 0.0, 3.2], [4.1, -1.2, -3.3, 0.5]], dtype=np.float64)
    rtol = None
    hermitian = False
    rcond = np.array(1e-12, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 3: 2D square symmetric matrix, float32, hermitian=True, rtol specified
    base = np.array([[2.0, -1.0], [1.0, 3.0]], dtype=np.float32)
    a = base @ base.T
    rtol = np.array(1e-6, dtype=np.float32)
    hermitian = True
    rcond = None
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 4: 2D complex Hermitian matrix, complex64, hermitian=True, rcond specified
    mat = np.array(
        [[2.0 + 0.0j, 1.0 - 2.0j], [1.0 + 2.0j, 5.0 + 0.0j]],
        dtype=np.complex64,
    )
    a = (mat + mat.conj().T) / 2.0
    rtol = None
    hermitian = True
    rcond = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 5: 2D general complex matrix, complex128, hermitian=False, rtol specified
    a = np.array(
        [
            [1.0 + 1.0j, 2.0 - 1.0j, 0.5 + 0.0j],
            [-1.5 + 2.0j, 0.0 - 3.0j, 4.0 + 1.0j],
        ],
        dtype=np.complex128,
    )
    rtol = np.array(1e-10, dtype=np.float64)
    hermitian = False
    rcond = None
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 6: 3D batched non-square matrices, float32, hermitian=False, rcond specified
    np.random.seed(42)
    a = np.random.randn(4, 5, 3).astype(np.float32)
    rtol = None
    hermitian = False
    rcond = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 7: 3D batched Hermitian matrices, float64, hermitian=True, rtol specified
    batch_base = np.random.randn(3, 4, 4).astype(np.float64)
    a = np.matmul(batch_base, np.swapaxes(batch_base, -1, -2))
    rtol = np.array(1e-8, dtype=np.float64)
    hermitian = True
    rcond = None
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 8: 4D batched matrices with batch rcond, float32, hermitian=False, rcond specified
    a = np.random.randn(2, 3, 4, 2).astype(np.float32)
    rtol = None
    hermitian = False
    rcond = np.full((2, 3), 1e-4, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 9: Rank-deficient matrix (outer product), float32, hermitian=False, rtol specified
    u = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    v = np.array([[1.0, -1.0, 2.0]], dtype=np.float32)
    a = u @ v
    rtol = np.array(1e-3, dtype=np.float32)
    hermitian = False
    rcond = None
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    # Input 10: 1x1 scalar-like matrix, float64, hermitian=True, rcond specified
    a = np.array([[42.0]], dtype=np.float64)
    rtol = None
    hermitian = True
    rcond = np.array(1e-15, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": rcond}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.pinv_2"] = pinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.pinv_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.pinv_2'.")


check_valid('jax.numpy.linalg.pinv', generated_inputs['jax.numpy.linalg.pinv_2'], lib="jax", suffix=2)
