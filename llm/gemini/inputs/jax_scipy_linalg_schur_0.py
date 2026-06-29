
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def schur_inputs():
    list_of_inputs = []

    # Input 1: Simple 3x3 real float32 matrix with "real" output
    a = np.array([[1.0, 2.0, 3.0],
                  [1.0, 4.0, 2.0],
                  [3.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same 3x3 real float32 matrix with "complex" output
    a = np.array([[1.0, 2.0, 3.0],
                  [1.0, 4.0, 2.0],
                  [3.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 real float64 matrix with "real" output
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4x4 real float64 matrix with "complex" output
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 matrix with "complex" output
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched (2, 2, 2) real float32 matrices with "real" output
    a = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched (3, 4, 4) real float64 matrices with "complex" output
    a = np.random.randn(3, 4, 4).astype(np.float64)
    input_dict = {"a": a, "output": "complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Symmetric real float32 matrix with "real" output
    a = np.random.randn(5, 5).astype(np.float32)
    a = a + a.T
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 10x10 real float32 matrix with negative values
    a = (np.random.randn(10, 10) * 10).astype(np.float32)
    input_dict = {"a": a, "output": "real"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched (2, 3, 3) complex128 matrices with "complex" output
    a = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    input_dict = {"a": a, "output": "complex"}
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
