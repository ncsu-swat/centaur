
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import jax.numpy as jnp
import numpy as np


class LUTuple(tuple):

    def __array__(self, *args, **kwargs):
        vals = []
        for x in self:
            arr = np.asanyarray(x)
            if arr.size > 0:
                vals.append(float(np.min(np.real(arr))))
                vals.append(float(np.max(np.real(arr))))
        return np.array(vals if vals else [0.0], dtype=np.float64)


def lu_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32, 1D RHS
    a1 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    lu1, piv1 = jax.scipy.linalg.lu_factor(jnp.array(a1))
    b1 = np.array([3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu1), np.asarray(piv1))),
            "b": b1,
            "trans": 0,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    # Input 2: 3x3 float64, 2D RHS
    np.random.seed(42)
    a2 = np.random.randn(3, 3).astype(np.float64)
    lu2, piv2 = jax.scipy.linalg.lu_factor(jnp.array(a2))
    b2 = np.random.randn(3, 2).astype(np.float64)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu2), np.asarray(piv2))),
            "b": b2,
            "trans": 0,
            "overwrite_b": False,
            "check_finite": False,
        }
    )

    # Input 3: 4x4 float32, transpose solve (trans = 1)
    a3 = np.random.randn(4, 4).astype(np.float32)
    lu3, piv3 = jax.scipy.linalg.lu_factor(jnp.array(a3))
    b3 = np.random.randn(4).astype(np.float32)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu3), np.asarray(piv3))),
            "b": b3,
            "trans": 1,
            "overwrite_b": True,
            "check_finite": True,
        }
    )

    # Input 4: 3x3 complex64, conjugate transpose solve (trans = 2)
    a4 = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    lu4, piv4 = jax.scipy.linalg.lu_factor(jnp.array(a4))
    b4 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu4), np.asarray(piv4))),
            "b": b4,
            "trans": 2,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    # Input 5: 5x5 complex128, trans = 2, 2D RHS
    a5 = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(
        np.complex128
    )
    lu5, piv5 = jax.scipy.linalg.lu_factor(jnp.array(a5))
    b5 = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu5), np.asarray(piv5))),
            "b": b5,
            "trans": 2,
            "overwrite_b": True,
            "check_finite": False,
        }
    )

    # Input 6: 1x1 float32
    a6 = np.array([[5.5]], dtype=np.float32)
    lu6, piv6 = jax.scipy.linalg.lu_factor(jnp.array(a6))
    b6 = np.array([11.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu6), np.asarray(piv6))),
            "b": b6,
            "trans": 0,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    # Input 7: Batched (2, 3, 3) float32, batched RHS (2, 3)
    a7 = np.random.randn(2, 3, 3).astype(np.float32)
    lu7, piv7 = jax.scipy.linalg.lu_factor(jnp.array(a7))
    b7 = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu7), np.asarray(piv7))),
            "b": b7,
            "trans": 0,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    # Input 8: Batched (4, 2, 2) float64, batched 2D RHS (4, 2, 1), trans = 1
    a8 = np.random.randn(4, 2, 2).astype(np.float64)
    lu8, piv8 = jax.scipy.linalg.lu_factor(jnp.array(a8))
    b8 = np.random.randn(4, 2, 1).astype(np.float64)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu8), np.asarray(piv8))),
            "b": b8,
            "trans": 1,
            "overwrite_b": False,
            "check_finite": False,
        }
    )

    # Input 9: 6x6 float32 with negative elements
    a9 = -np.abs(np.random.randn(6, 6).astype(np.float32)) - 2.0 * np.eye(
        6, dtype=np.float32
    )
    lu9, piv9 = jax.scipy.linalg.lu_factor(jnp.array(a9))
    b9 = -np.abs(np.random.randn(6).astype(np.float32))
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu9), np.asarray(piv9))),
            "b": b9,
            "trans": 0,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    # Input 10: Batched (2, 2, 3, 3) float32, batched RHS (2, 2, 3)
    a10 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    lu10, piv10 = jax.scipy.linalg.lu_factor(jnp.array(a10))
    b10 = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu10), np.asarray(piv10))),
            "b": b10,
            "trans": 0,
            "overwrite_b": True,
            "check_finite": True,
        }
    )

    # Input 11: 4x4 float64, matrix RHS (4, 4), trans = 1
    a11 = np.random.randn(4, 4).astype(np.float64)
    lu11, piv11 = jax.scipy.linalg.lu_factor(jnp.array(a11))
    b11 = np.eye(4, dtype=np.float64)
    list_of_inputs.append(
        {
            "lu_and_piv": LUTuple((np.asarray(lu11), np.asarray(piv11))),
            "b": b11,
            "trans": 1,
            "overwrite_b": False,
            "check_finite": True,
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.lu_solve'.")


check_valid('jax.scipy.linalg.lu_solve', generated_inputs['jax.scipy.linalg.lu_solve'], lib="jax", suffix=0)
