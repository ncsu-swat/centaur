
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: 2x2 matrix with power 0
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n = int(0)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 2: 2x2 matrix with power 1
    a = np.array([[2.0, -1.0], [1.0, 3.0]], dtype=np.float32)
    n = int(1)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 3: 3x3 matrix with power 3
    a = np.array(
        [[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float32
    )
    n = int(3)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 4: 2x2 invertible matrix with negative power -1
    a = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)
    n = int(-1)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 5: 2x2 invertible matrix with negative power -2
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    n = int(-2)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 6: 1x1 matrix with power 5
    a = np.array([[2.5]], dtype=np.float32)
    n = int(5)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 7: 1x1 matrix with negative power -3
    a = np.array([[2.0]], dtype=np.float64)
    n = int(-3)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 8: Batched 3D matrix (batch size 4, 2x2) with power 2
    np.random.seed(42)
    a = np.random.randn(4, 2, 2).astype(np.float32)
    n = int(2)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 9: Batched 3D invertible matrix (batch size 3, 3x3) with negative power -1
    diag_dominance = 10.0 * np.eye(3, dtype=np.float64)
    a = np.stack(
        [
            diag_dominance + np.random.randn(3, 3).astype(np.float64)
            for _ in range(3)
        ]
    )
    n = int(-1)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 10: 4D batched matrix (2, 2, 3, 3) with power 4
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    n = int(4)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 11: Complex 2x2 matrix with power 2
    a = (
        np.array([[1.0 + 2.0j, -1.0j], [3.0, 2.0 - 1.0j]], dtype=np.complex64)
    )
    n = int(2)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    # Input 12: Integer matrix with positive power
    a = np.array([[1, 2], [0, 1]], dtype=np.int32)
    n = int(5)
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_power'.")


check_valid('jax.numpy.linalg.matrix_power', generated_inputs['jax.numpy.linalg.matrix_power'], lib="jax", suffix=0)
