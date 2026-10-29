
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def schur_inputs():
    list_of_inputs = []

    # Input 1: 2x2 real float32, real output
    a = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 real float32, complex output
    a = np.array(
        [[1.0, 2.0, 3.0], [-1.0, 4.0, -2.0], [3.0, 2.0, 1.0]], dtype=np.float32
    )
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 real float64, real output
    np.random.seed(0)
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 5x5 real float64, complex output
    np.random.seed(1)
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D array (2, 3, 3) real float32, real output
    np.random.seed(2)
    a = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched 3D array (3, 4, 4) real float64, complex output
    np.random.seed(3)
    a = np.random.randn(3, 4, 4).astype(np.float64)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3x3 complex64, complex output
    np.random.seed(4)
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4x4 complex128, complex output
    np.random.seed(5)
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex128
    )
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batched 3D array (2, 2, 2) complex64, complex output
    np.random.seed(6)
    a = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(
        np.complex64
    )
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched 4D array (2, 2, 3, 3) real float32, real output
    np.random.seed(7)
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1x1 real float32, real output
    a = np.array([[-3.5]], dtype=np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.schur"] = schur_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.schur' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.schur'.")


check_valid('jax.scipy.linalg.schur', generated_inputs['jax.scipy.linalg.schur'], lib="jax", suffix=0)
