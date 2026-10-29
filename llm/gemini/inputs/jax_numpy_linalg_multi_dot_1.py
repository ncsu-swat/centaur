
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multi_dot_inputs():
    list_of_inputs = []

    # Input 1: 3 square matrices of size 4x4, float32, 'default' precision
    arrays1 = np.random.randn(3, 4, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays1, "precision": "default"})
    )

    # Input 2: 2 square matrices of size 8x8, float64, 'high' precision
    arrays2 = np.random.randn(2, 8, 8).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"arrays": arrays2, "precision": "high"}))

    # Input 3: 4 square matrices of size 3x3, float32, 'highest' precision
    arrays3 = np.random.randn(4, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays3, "precision": "highest"})
    )

    # Input 4: 5 square matrices of size 2x2, float64, 'fastest' precision
    arrays4 = np.random.randn(5, 2, 2).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays4, "precision": "fastest"})
    )

    # Input 5: 2 complex64 matrices of size 5x5, 'high' precision
    arrays5 = (
        np.random.randn(2, 5, 5) + 1j * np.random.randn(2, 5, 5)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"arrays": arrays5, "precision": "high"}))

    # Input 6: 3 complex128 matrices of size 6x6, 'highest' precision
    arrays6 = (
        np.random.randn(3, 6, 6) + 1j * np.random.randn(3, 6, 6)
    ).astype(np.complex128)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays6, "precision": "highest"})
    )

    # Input 7: 6 square matrices of size 2x2 with negative values, float32, 'default' precision
    arrays7 = -np.abs(np.random.randn(6, 2, 2).astype(np.float32))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays7, "precision": "default"})
    )

    # Input 8: 4 square matrices of size 5x5, float64, 'float32' precision
    arrays8 = np.random.randn(4, 5, 5).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays8, "precision": "float32"})
    )

    # Input 9: 3 square matrices of size 10x10, float32, 'highest' precision
    arrays9 = np.random.randn(3, 10, 10).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays9, "precision": "highest"})
    )

    # Input 10: 7 square matrices of size 3x3, float32, 'default' precision
    arrays10 = np.random.randn(7, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays10, "precision": "default"})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.multi_dot_1"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.multi_dot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.multi_dot_1'.")


check_valid('jax.numpy.linalg.multi_dot', generated_inputs['jax.numpy.linalg.multi_dot_1'], lib="jax", suffix=1)
