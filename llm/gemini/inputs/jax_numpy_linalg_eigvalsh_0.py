
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eigvalsh_inputs():
    list_of_inputs = []

    def make_symmetric(shape, dtype=np.float32):
        if np.issubdtype(dtype, np.complexfloating):
            if dtype == np.complex64:
                real_dtype = np.float32
            else:
                real_dtype = np.float64
            a = np.random.randn(*shape).astype(real_dtype) + 1j * np.random.randn(*shape).astype(real_dtype)
            return (a + np.swapaxes(a.conj(), -1, -2)).astype(dtype)
        else:
            a = np.random.randn(*shape).astype(dtype)
            return a + np.swapaxes(a, -1, -2)

    # Input 1: Simple 2x2 real, UPLO='L', symmetrize_input=True
    a = make_symmetric((2, 2), np.float32)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 2x2 real, UPLO='U', symmetrize_input=False
    a = make_symmetric((2, 2), np.float32)
    input_dict = {"a": a, "UPLO": "U", "symmetrize_input": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 complex, UPLO='L', symmetrize_input=True
    a = make_symmetric((3, 3), np.complex64)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4x4 complex double precision, UPLO='U', symmetrize_input=True
    a = make_symmetric((4, 4), np.complex128)
    input_dict = {"a": a, "UPLO": "U", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 5x5 real double precision, UPLO='L', symmetrize_input=False
    a = make_symmetric((5, 5), np.float64)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched 3D real, UPLO='U', symmetrize_input=True
    a = make_symmetric((3, 4, 4), np.float32)
    input_dict = {"a": a, "UPLO": "U", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dim batch 4D real, UPLO='L', symmetrize_input=True
    a = make_symmetric((2, 2, 3, 3), np.float64)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched 3D complex, UPLO='U', symmetrize_input=False
    a = make_symmetric((2, 5, 5), np.complex64)
    input_dict = {"a": a, "UPLO": "U", "symmetrize_input": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 matrix (edge case), UPLO='L', symmetrize_input=True
    a = make_symmetric((1, 1), np.float32)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 50x50 matrix, UPLO='L', symmetrize_input=False
    a = make_symmetric((50, 50), np.float32)
    input_dict = {"a": a, "UPLO": "L", "symmetrize_input": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.eigvalsh"] = eigvalsh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eigvalsh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eigvalsh'.")


check_valid('jax.numpy.linalg.eigvalsh', generated_inputs['jax.numpy.linalg.eigvalsh'], lib="jax", suffix=0)
