
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 square matrix, positive power
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n = 3
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 2: float64 square matrix, negative power (requires invertible matrix)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    n = -2
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 3: Large float32 matrix, zero power
    a = np.random.randn(10, 10).astype(np.float32)
    n = 0
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 4: Batched float32 matrices, positive power
    a = np.random.randn(3, 4, 4).astype(np.float32)
    n = 5
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 5: Complex64 square matrix, positive power
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    n = 4
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 6: float16 square matrix, small positive power
    a = np.random.randn(2, 2).astype(np.float16)
    n = 2
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 7: 1x1 matrix, large positive power
    a = np.array([[2.5]], dtype=np.float32)
    n = 10
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 8: High-dimensional batched float64 matrices, positive power
    a = np.random.randn(2, 2, 3, 3).astype(np.float64)
    n = 3
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 9: float32 invertible matrix (diagonal), negative power
    a = np.diag([2.0, 3.0, 4.0]).astype(np.float32)
    n = -3
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 10: 5x5 matrix, positive power
    a = np.random.randn(5, 5).astype(np.float32)
    n = 6
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

    # Input 11: Batched float32 matrices, zero power
    a = np.random.randn(2, 5, 5).astype(np.float32)
    n = 0
    list_of_inputs.append({"a": copy.deepcopy(a), "n": n})

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
