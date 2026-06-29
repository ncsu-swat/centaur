
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def slogdet_inputs():
    list_of_inputs = []

    # Input 1: float32, 2x2 matrix, LU method
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 3x3 matrix with zero elements, QR method
    a = np.array([[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    input_dict = {"a": a, "method": "qr"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 4x4 matrix, LU method
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 5x5 matrix, QR method
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"a": a, "method": "qr"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, batched 3D array (2, 3, 3), LU method
    a = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, batched 3D array (4, 2, 2), QR method
    a = np.random.randn(4, 2, 2).astype(np.float64)
    input_dict = {"a": a, "method": "qr"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 1x1 matrix, LU method
    a = np.array([[5.0]], dtype=np.float32)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, 1x1 matrix with a negative value, QR method
    a = np.array([[-3.5]], dtype=np.float64)
    input_dict = {"a": a, "method": "qr"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, higher dimension batch (2, 2, 4, 4), LU method
    a = np.random.randn(2, 2, 4, 4).astype(np.float32)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, larger 10x10 matrix, QR method
    a = np.random.randn(10, 10).astype(np.float64)
    input_dict = {"a": a, "method": "qr"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: complex64, 2x2 complex matrix, LU method
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {"a": a, "method": "lu"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
