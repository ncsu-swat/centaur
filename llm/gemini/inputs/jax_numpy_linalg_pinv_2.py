
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pinv_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float32 matrix
    a = np.random.randn(3, 5).astype(np.float32)
    hermitian = False
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square symmetric float32 matrix with hermitian=True
    a = np.random.randn(4, 4).astype(np.float32)
    a = a + a.T
    hermitian = True
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batched 3D float32 matrices with rtol as a 0D tensor
    a = np.random.randn(2, 3, 4).astype(np.float32)
    rtol = np.array(1e-5, dtype=np.float32)
    hermitian = False
    input_dict = {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 matrix with rcond as a 0D tensor
    a = np.random.randn(5, 3).astype(np.float64)
    rcond = np.array(1e-6, dtype=np.float64)
    hermitian = False
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched square symmetric matrices
    a = np.random.randn(2, 3, 3).astype(np.float32)
    a = a + a.transpose(0, 2, 1)
    hermitian = True
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 matrix
    a = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    hermitian = False
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128 Hermitian matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex128)
    a = a + a.conj().T
    hermitian = True
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched matrices with a multi-dimensional rtol tensor
    a = np.random.randn(3, 2, 4, 4).astype(np.float32)
    rtol = np.random.uniform(1e-6, 1e-4, size=(3, 2)).astype(np.float32)
    hermitian = False
    input_dict = {"a": a, "rtol": rtol, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Square float32 matrix with rcond tensor
    a = np.random.randn(6, 6).astype(np.float32)
    rcond = np.array(1e-5, dtype=np.float32)
    hermitian = False
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional float32 matrix
    a = np.random.randn(10, 2).astype(np.float32)
    hermitian = False
    input_dict = {"a": a, "rtol": None, "hermitian": hermitian, "rcond": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.pinv_2"] = pinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.pinv_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.pinv_2'.")


check_valid('jax.numpy.linalg.pinv', generated_inputs['jax.numpy.linalg.pinv_2'], lib="jax", suffix=2)
