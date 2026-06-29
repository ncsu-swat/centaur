
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eigh_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 symmetric, UPLO='L', symmetrize_input=True
    a = np.random.randn(4, 4).astype(np.float32)
    a = a + a.T
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 symmetric, UPLO='U', symmetrize_input=False
    a = np.random.randn(3, 3).astype(np.float32)
    a = a + a.T
    input_dict = {
        "a": a,
        "UPLO": "U",
        "symmetrize_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 symmetric, UPLO='L', symmetrize_input=True
    a = np.random.randn(5, 5).astype(np.float64)
    a = a + a.T
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex64 Hermitian, UPLO='L', symmetrize_input=True
    a_real = np.random.randn(4, 4).astype(np.float32)
    a_imag = np.random.randn(4, 4).astype(np.float32)
    a = a_real + 1j * a_imag
    a = a + a.conj().T
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex128 Hermitian, UPLO='U', symmetrize_input=True
    a_real = np.random.randn(3, 3).astype(np.float64)
    a_imag = np.random.randn(3, 3).astype(np.float64)
    a = a_real + 1j * a_imag
    a = a + a.conj().T
    input_dict = {
        "a": a,
        "UPLO": "U",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched float32 symmetric, UPLO='L', symmetrize_input=True
    a = np.random.randn(2, 6, 6).astype(np.float32)
    a = a + np.swapaxes(a, -1, -2)
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched float64 symmetric, UPLO='U', symmetrize_input=False
    a = np.random.randn(2, 3, 4, 4).astype(np.float64)
    a = a + np.swapaxes(a, -1, -2)
    input_dict = {
        "a": a,
        "UPLO": "U",
        "symmetrize_input": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 non-symmetric with symmetrize_input=True
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D batched complex64 Hermitian, UPLO='L', symmetrize_input=True
    a_real = np.random.randn(2, 4, 4).astype(np.float32)
    a_imag = np.random.randn(2, 4, 4).astype(np.float32)
    a = a_real + 1j * a_imag
    a = a + np.swapaxes(a, -1, -2).conj()
    input_dict = {
        "a": a,
        "UPLO": "L",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 symmetric with large negative values, UPLO='U', symmetrize_input=True
    a = np.random.randn(10, 10).astype(np.float64) * -10.0
    a = a + a.T
    input_dict = {
        "a": a,
        "UPLO": "U",
        "symmetrize_input": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.eigh"] = eigh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eigh'.")


check_valid('jax.numpy.linalg.eigh', generated_inputs['jax.numpy.linalg.eigh'], lib="jax", suffix=0)
