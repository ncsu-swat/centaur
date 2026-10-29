
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sqrtm_inputs():
    list_of_inputs = []

    # Input 1: 3x3 float32 matrix
    A = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 2.0], [3.0, 2.0, 1.0]], dtype=np.float32
    )
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 2: 2x2 float64 matrix
    A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 3: 1x1 float32 matrix
    A = np.array([[9.0]], dtype=np.float32)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 4: 4x4 complex64 matrix
    A = np.array(
        [
            [1.0 + 2.0j, 0.0, 1.0j, 0.5],
            [0.0, 2.0 - 1.0j, 0.0, 1.0],
            [1.0j, 0.0, 3.0, 0.0],
            [0.5, 1.0, 0.0, 4.0 + 1.0j],
        ],
        dtype=np.complex64,
    )
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 5: 4x4 complex128 matrix
    rng = np.random.RandomState(42)
    A_real = rng.randn(4, 4)
    A_imag = rng.randn(4, 4)
    A = (A_real + 1j * A_imag).astype(np.complex128)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 6: 2x2 matrix with negative values
    A = np.array([[-5.0, 2.0], [1.0, -3.0]], dtype=np.float32)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 7: 5x5 symmetric positive definite float32 matrix
    M = rng.randn(5, 5).astype(np.float32)
    A = M @ M.T + 0.1 * np.eye(5, dtype=np.float32)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 8: 3x3 upper triangular matrix
    A = np.array(
        [[4.0, 1.0, -2.0], [0.0, 9.0, 3.0], [0.0, 0.0, 16.0]], dtype=np.float64
    )
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 9: 6x6 float64 random matrix
    A = rng.randn(6, 6).astype(np.float64)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 10: 2x2 identity matrix float32
    A = np.eye(2, dtype=np.float32)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    # Input 11: 3x3 diagonal matrix with negative elements
    A = np.diag([-4.0, -9.0, 16.0]).astype(np.float32)
    list_of_inputs.append({"A": copy.deepcopy(A), "blocksize": 1})

    return list_of_inputs


generated_inputs["jax.scipy.linalg.sqrtm"] = sqrtm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.sqrtm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.sqrtm'.")


check_valid('jax.scipy.linalg.sqrtm', generated_inputs['jax.scipy.linalg.sqrtm'], lib="jax", suffix=0)
