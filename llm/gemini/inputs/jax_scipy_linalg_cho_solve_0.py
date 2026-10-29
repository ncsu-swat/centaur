
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class ChoSolveTuple(tuple):

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return super().__new__(cls, args[0])
        return super().__new__(cls, args)

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self[0])
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        if copy is not None:
            return np.array(arr, dtype=dtype, copy=copy)
        return np.asarray(arr, dtype=dtype)


def cho_solve_inputs():
    list_of_inputs = []

    # Input 1: 1D real float32, lower=True
    c1 = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    b1 = np.array([1.0, -2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c1, True)),
                "b": b1,
                "overwrite_b": False,
                "check_finite": True,
            }
        )
    )

    # Input 2: 1D real float64, upper=True (lower=False)
    a2 = np.array(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], dtype=np.float64
    )
    c2 = np.linalg.cholesky(a2).T
    b2 = np.array([2.5, -1.0, 3.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c2, False)),
                "b": b2,
                "overwrite_b": True,
                "check_finite": False,
            }
        )
    )

    # Input 3: 2D b, float32, lower=True
    a3 = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    c3 = np.linalg.cholesky(a3)
    b3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c3, True)),
                "b": b3,
                "overwrite_b": False,
                "check_finite": False,
            }
        )
    )

    # Input 4: 2D b with multiple columns, float64, lower=False
    a4 = np.array(
        [
            [5.0, -1.0, 0.0, 0.2],
            [-1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, -0.5],
            [0.2, 0.0, -0.5, 2.0],
        ],
        dtype=np.float64,
    )
    c4 = np.linalg.cholesky(a4).T
    b4 = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 2.0],
            [-1.0, 0.5, 1.5],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c4, False)),
                "b": b4,
                "overwrite_b": True,
                "check_finite": True,
            }
        )
    )

    # Input 5: 1x1 matrix, float32, lower=True
    c5 = np.array([[3.5]], dtype=np.float32)
    b5 = np.array([7.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c5, True)),
                "b": b5,
                "overwrite_b": False,
                "check_finite": True,
            }
        )
    )

    # Input 6: 2D b (column vector), float32, lower=False
    c6 = np.array(
        [[2.0, 0.5, -0.2], [0.0, 1.5, 0.3], [0.0, 0.0, 2.5]], dtype=np.float32
    )
    b6 = np.array([[1.0], [-2.0], [0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c6, False)),
                "b": b6,
                "overwrite_b": False,
                "check_finite": False,
            }
        )
    )

    # Input 7: Batched inputs (batch size 2), float32, lower=True
    a7_1 = np.array([[2.0, 0.5], [0.5, 2.0]], dtype=np.float32)
    a7_2 = np.array([[3.0, -0.5], [-0.5, 3.0]], dtype=np.float32)
    c7 = np.stack([np.linalg.cholesky(a7_1), np.linalg.cholesky(a7_2)], axis=0)
    b7 = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c7, True)),
                "b": b7,
                "overwrite_b": False,
                "check_finite": True,
            }
        )
    )

    # Input 8: Batched inputs with 2D right-hand sides, float64, lower=False
    a8_1 = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    a8_2 = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    a8_3 = np.array([[3.0, -1.0], [-1.0, 2.0]], dtype=np.float64)
    c8 = np.stack(
        [
            np.linalg.cholesky(a8_1).T,
            np.linalg.cholesky(a8_2).T,
            np.linalg.cholesky(a8_3).T,
        ],
        axis=0,
    )
    b8 = np.ones((3, 2, 4), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c8, False)),
                "b": b8,
                "overwrite_b": True,
                "check_finite": False,
            }
        )
    )

    # Input 9: Higher-dimensional batched inputs (2, 2, 2, 2), float32, lower=True
    base_l9 = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    c9 = np.tile(base_l9, (2, 2, 1, 1))
    b9 = np.ones((2, 2, 2, 1), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c9, True)),
                "b": b9,
                "overwrite_b": False,
                "check_finite": True,
            }
        )
    )

    # Input 10: 4x4 matrix, float64, lower=False
    c10 = np.array(
        [
            [3.0, 1.0, 0.5, -0.5],
            [0.0, 2.0, 0.2, 0.1],
            [0.0, 0.0, 2.5, 0.3],
            [0.0, 0.0, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    b10 = np.array([-3.0, 1.5, 0.0, 4.2], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "c_and_lower": ChoSolveTuple((c10, False)),
                "b": b10,
                "overwrite_b": False,
                "check_finite": False,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.cho_solve"] = cho_solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cho_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cho_solve'.")


check_valid('jax.scipy.linalg.cho_solve', generated_inputs['jax.scipy.linalg.cho_solve'], lib="jax", suffix=0)
