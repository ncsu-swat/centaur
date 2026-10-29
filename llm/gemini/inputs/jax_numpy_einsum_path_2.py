
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_path_inputs():
    list_of_inputs = []

    # Input 1: 2D matrix diagonal extraction / trace
    x = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"subscripts": "ii", "operands": x, "optimize": "auto"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrix transpose
    x = np.random.randn(4, 7).astype(np.float64)
    input_dict = {"subscripts": "ij->ji", "operands": x, "optimize": "optimal"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array sum reduction with negative values
    x = np.arange(-10, 10, dtype=np.int32)
    input_dict = {"subscripts": "i->", "operands": x, "optimize": "greedy"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor permutation
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"subscripts": "ijk->kji", "operands": x, "optimize": "eager"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D diagonal extraction
    x = np.random.randint(-5, 5, size=(6, 6)).astype(np.int64)
    input_dict = {"subscripts": "ii->i", "operands": x, "optimize": "auto"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D row summation
    x = np.random.randn(8, 12).astype(np.float16)
    input_dict = {"subscripts": "ij->i", "operands": x, "optimize": "greedy"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor axis swap
    x = np.random.randn(2, 4, 3, 5).astype(np.float32)
    input_dict = {"subscripts": "ijkl->ljki", "operands": x, "optimize": "optimal"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D identity / copy with complex values
    x = (np.random.randn(10) + 1j * np.random.randn(10)).astype(np.complex64)
    input_dict = {"subscripts": "i->i", "operands": x, "optimize": "auto"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor reduction along two axes
    x = np.random.randn(3, 5, 7).astype(np.float64)
    input_dict = {"subscripts": "ijk->i", "operands": x, "optimize": "optimal"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D matrix trace reduction to scalar
    x = np.random.randn(10, 10).astype(np.float32)
    input_dict = {"subscripts": "ii->", "operands": x, "optimize": "greedy"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D tensor reduction to 2D
    x = np.random.randn(3, 4, 6, 5).astype(np.float32)
    input_dict = {"subscripts": "ijkl->ik", "operands": x, "optimize": "auto"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 3D tensor reduction along first axis
    x = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {"subscripts": "ijk->jk", "operands": x, "optimize": "greedy"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.einsum_path_2"] = einsum_path_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_path_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_path_2'.")


check_valid('jax.numpy.einsum_path', generated_inputs['jax.numpy.einsum_path_2'], lib="jax", suffix=2)
