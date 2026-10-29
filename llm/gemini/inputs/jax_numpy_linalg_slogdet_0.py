
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def slogdet_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix float32, method 'lu'
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 2: 2D square matrix float32, method 'qr'
    a = np.array([[2.0, -1.0], [1.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "method": "qr"})

    # Input 3: 2D square matrix float64 with negative determinant
    a = np.array(
        [[1.0, 2.0, 3.0], [0.0, -4.0, 5.0], [6.0, 0.0, 1.0]], dtype=np.float64
    )
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 4: 2D complex64 matrix, method 'lu'
    a = np.array([[1.0 + 2.0j, -1.0j], [0.5, 2.0 - 1.0j]], dtype=np.complex64)
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 5: 2D complex128 matrix, method 'lu'
    a = np.array(
        [[2.0 + 1.0j, 0.0, -1.0j], [1.0j, 1.0 - 1.0j, 2.0], [0.0, 3.0j, -1.0]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 6: 3D batched square matrices float32, method 'lu'
    a = np.random.randn(3, 4, 4).astype(np.float32)
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 7: 3D batched square matrices float64, method 'qr'
    a = np.random.randn(5, 2, 2).astype(np.float64)
    list_of_inputs.append({"a": a, "method": "qr"})

    # Input 8: 4D batched matrices float32, method 'lu'
    a = np.random.randn(2, 3, 3, 3).astype(np.float32)
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 9: 1x1 scalar-like matrix float32, method 'lu'
    a = np.array([[-5.5]], dtype=np.float32)
    list_of_inputs.append({"a": a, "method": "lu"})

    # Input 10: 5x5 Identity matrix float32, method 'qr'
    a = np.eye(5, dtype=np.float32)
    list_of_inputs.append({"a": a, "method": "qr"})

    # Input 11: 4x4 matrix with negative and zero values float64, method 'qr'
    a = np.array(
        [
            [-2.0, 0.0, 1.0, 4.0],
            [1.0, -3.0, 0.0, -1.0],
            [0.0, 2.0, -1.0, 3.0],
            [-1.0, 4.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append({"a": a, "method": "qr"})

    # Input 12: 5D batched square matrices float32, method 'qr'
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "method": "qr"})

    return list_of_inputs


generated_inputs["jax.numpy.linalg.slogdet"] = slogdet_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.slogdet'.")


check_valid('jax.numpy.linalg.slogdet', generated_inputs['jax.numpy.linalg.slogdet'], lib="jax", suffix=0)
