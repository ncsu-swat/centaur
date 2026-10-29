
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import scipy.linalg


def rsf2csf_inputs():
    list_of_inputs = []

    # Helper function to get valid real Schur form and transformation matrix
    def get_schur(A):
        return scipy.linalg.schur(A, output="real")

    # Input 1: Basic 3x3 matrix with float64
    A1 = np.array(
        [[0.0, 3.0, 3.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=np.float64
    )
    T1, Z1 = get_schur(A1)
    list_of_inputs.append(
        {"T": T1.astype(np.float64), "Z": Z1.astype(np.float64), "check_finite": True}
    )

    # Input 2: 2x2 matrix with complex conjugate eigenvalues (float32)
    A2 = np.array([[0.0, -2.0], [2.0, 0.0]], dtype=np.float32)
    T2, Z2 = get_schur(A2)
    list_of_inputs.append(
        {
            "T": T2.astype(np.float32),
            "Z": Z2.astype(np.float32),
            "check_finite": False,
        }
    )

    # Input 3: 1x1 trivial matrix
    A3 = np.array([[5.0]], dtype=np.float32)
    T3, Z3 = get_schur(A3)
    list_of_inputs.append(
        {"T": T3.astype(np.float32), "Z": Z3.astype(np.float32), "check_finite": True}
    )

    # Input 4: 4x4 matrix with negative and mixed values (float64)
    rng = np.random.RandomState(42)
    A4 = rng.randn(4, 4).astype(np.float64)
    T4, Z4 = get_schur(A4)
    list_of_inputs.append(
        {"T": T4.astype(np.float64), "Z": Z4.astype(np.float64), "check_finite": True}
    )

    # Input 5: 2x2 already upper triangular matrix
    T5 = np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    Z5 = np.eye(2, dtype=np.float32)
    list_of_inputs.append(
        {
            "T": T5.astype(np.float32),
            "Z": Z5.astype(np.float32),
            "check_finite": False,
        }
    )

    # Input 6: 5x5 random matrix
    A6 = rng.randn(5, 5).astype(np.float64)
    T6, Z6 = get_schur(A6)
    list_of_inputs.append(
        {"T": T6.astype(np.float64), "Z": Z6.astype(np.float64), "check_finite": True}
    )

    # Input 7: Batched 3D tensor (shape: 3, 3, 3)
    T_list7, Z_list7 = [], []
    for _ in range(3):
        t, z = get_schur(rng.randn(3, 3))
        T_list7.append(t)
        Z_list7.append(z)
    T7 = np.stack(T_list7, axis=0).astype(np.float32)
    Z7 = np.stack(Z_list7, axis=0).astype(np.float32)
    list_of_inputs.append(
        {"T": T7.astype(np.float32), "Z": Z7.astype(np.float32), "check_finite": True}
    )

    # Input 8: Batched 4D tensor (shape: 2, 2, 4, 4)
    T_list8, Z_list8 = [], []
    for _ in range(4):
        t, z = get_schur(rng.randn(4, 4))
        T_list8.append(t)
        Z_list8.append(z)
    T8 = np.stack(T_list8, axis=0).reshape(2, 2, 4, 4).astype(np.float64)
    Z8 = np.stack(Z_list8, axis=0).reshape(2, 2, 4, 4).astype(np.float64)
    list_of_inputs.append(
        {
            "T": T8.astype(np.float64),
            "Z": Z8.astype(np.float64),
            "check_finite": False,
        }
    )

    # Input 9: Broadcasted batch dimensions (T is (4, 3, 3), Z is (1, 3, 3))
    t_single, z_single = get_schur(rng.randn(3, 3))
    T9 = np.tile(t_single[None, ...], (4, 1, 1)).astype(np.float32)
    Z9 = z_single[None, ...].astype(np.float32)
    list_of_inputs.append(
        {"T": T9.astype(np.float32), "Z": Z9.astype(np.float32), "check_finite": True}
    )

    # Input 10: 6x6 matrix with float32
    A10 = rng.randn(6, 6).astype(np.float32)
    T10, Z10 = get_schur(A10)
    list_of_inputs.append(
        {
            "T": T10.astype(np.float32),
            "Z": Z10.astype(np.float32),
            "check_finite": False,
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.rsf2csf"] = rsf2csf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.rsf2csf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.rsf2csf'.")


check_valid('jax.scipy.linalg.rsf2csf', generated_inputs['jax.scipy.linalg.rsf2csf'], lib="jax", suffix=0)
