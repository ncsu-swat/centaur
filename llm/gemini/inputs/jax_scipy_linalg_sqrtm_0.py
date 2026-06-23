
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sqrtm_inputs():
    list_of_inputs = []

    # All inputs use shape (2, 2) and float32 to trigger JAX compilation only once.
    # Input 1: Identity matrix
    A1 = np.eye(2, dtype=np.float32)
    list_of_inputs.append({"A": A1, "blocksize": 1})

    # Input 2: Diagonal positive
    A2 = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    list_of_inputs.append({"A": A2, "blocksize": 1})

    # Input 3: Diagonal negative
    A3 = np.array([[-1.0, 0.0], [0.0, -4.0]], dtype=np.float32)
    list_of_inputs.append({"A": A3, "blocksize": 1})

    # Input 4: Symmetric positive definite
    A4 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"A": A4, "blocksize": 1})

    # Input 5: Upper triangular
    A5 = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"A": A5, "blocksize": 1})

    # Input 6: Lower triangular
    A6 = np.array([[4.0, 0.0], [3.0, 9.0]], dtype=np.float32)
    list_of_inputs.append({"A": A6, "blocksize": 1})

    # Input 7: Small values
    A7 = np.array([[1e-3, 0.0], [0.0, 1e-3]], dtype=np.float32)
    list_of_inputs.append({"A": A7, "blocksize": 1})

    # Input 8: Zero matrix
    A8 = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append({"A": A8, "blocksize": 1})

    # Input 9: Non-symmetric
    A9 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"A": A9, "blocksize": 1})

    # Input 10: Another non-symmetric
    A10 = np.array([[5.0, -1.0], [2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"A": A10, "blocksize": 1})

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
