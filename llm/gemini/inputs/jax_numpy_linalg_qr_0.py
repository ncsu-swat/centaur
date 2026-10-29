
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def qr_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32, mode="reduced"
    a = np.array(
        [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]],
        dtype=np.float32,
    )
    list_of_inputs.append({"a": a, "mode": "reduced"})

    # Input 2: 2D tall matrix (M > N), float64, mode="complete"
    a = np.random.randn(6, 3).astype(np.float64)
    list_of_inputs.append({"a": a, "mode": "complete"})

    # Input 3: 2D fat matrix (M < N), float32, mode="r"
    a = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"a": a, "mode": "r"})

    # Input 4: 2D matrix, float32, mode="raw"
    a = np.random.randn(5, 4).astype(np.float32)
    list_of_inputs.append({"a": a, "mode": "raw"})

    # Input 5: 3D batched square matrix, float32, mode="reduced"
    a = np.random.randn(2, 4, 4).astype(np.float32)
    list_of_inputs.append({"a": a, "mode": "reduced"})

    # Input 6: 3D batched tall matrix, complex64, mode="complete"
    real_part = np.random.randn(3, 5, 2).astype(np.float32)
    imag_part = np.random.randn(3, 5, 2).astype(np.float32)
    a = real_part + 1j * imag_part
    list_of_inputs.append({"a": a, "mode": "complete"})

    # Input 7: 4D batched matrix, float64, mode="r"
    a = np.random.randn(2, 2, 4, 3).astype(np.float64)
    list_of_inputs.append({"a": a, "mode": "r"})

    # Input 8: 2D complex128 matrix, mode="reduced"
    real_part = np.random.randn(4, 4).astype(np.float64)
    imag_part = np.random.randn(4, 4).astype(np.float64)
    a = real_part + 1j * imag_part
    list_of_inputs.append({"a": a, "mode": "reduced"})

    # Input 9: 3D batched matrix, float32, mode="raw"
    a = np.random.randn(4, 3, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "mode": "raw"})

    # Input 10: 2D tall matrix with negative and zero values, float32, mode="reduced"
    a = np.array(
        [[0.0, -1.5], [-2.0, 3.5], [4.0, 0.0], [-5.5, -6.0]], dtype=np.float32
    )
    list_of_inputs.append({"a": a, "mode": "reduced"})

    # Input 11: 3D batched fat matrix, float64, mode="complete"
    a = np.random.randn(2, 3, 6).astype(np.float64)
    list_of_inputs.append({"a": a, "mode": "complete"})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.linalg.qr"] = qr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.qr'.")


check_valid('jax.numpy.linalg.qr', generated_inputs['jax.numpy.linalg.qr'], lib="jax", suffix=0)
