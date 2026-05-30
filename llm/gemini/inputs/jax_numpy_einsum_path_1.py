
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_einsum_path_inputs():
    list_of_inputs = []

    # Input 1: Transpose operation (2D) with float32
    input_dict = {
        "subscripts": "ij->ji",
        "operands": np.random.randn(3, 4).astype(np.float32),
        "optimize": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Diagonal extraction with float32
    input_dict = {
        "subscripts": "ii->i",
        "operands": np.random.randn(5, 5).astype(np.float32),
        "optimize": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Permutation of 3D tensor with float64
    input_dict = {
        "subscripts": "ijk->kij",
        "operands": np.random.randn(2, 3, 4).astype(np.float64),
        "optimize": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Trace with int32
    input_dict = {
        "subscripts": "ii",
        "operands": np.random.randint(-10, 10, size=(10, 10)).astype(np.int32),
        "optimize": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sum reduction of 1D vector with float32
    input_dict = {
        "subscripts": "i->",
        "operands": np.random.randn(100).astype(np.float32),
        "optimize": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sum reduction along columns with float32
    input_dict = {
        "subscripts": "ij->i",
        "operands": np.random.randn(10, 20).astype(np.float32),
        "optimize": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Sum reduction of 3D tensor with float64
    input_dict = {
        "subscripts": "ijk->",
        "operands": np.random.randn(2, 2, 2).astype(np.float64),
        "optimize": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: No-op on a 1D vector with int32
    input_dict = {
        "subscripts": "i",
        "operands": np.random.randint(-5, 5, size=(5,)).astype(np.int32),
        "optimize": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ellipsis transpose with float32
    input_dict = {
        "subscripts": "ij...->ji...",
        "operands": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "optimize": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Ellipsis diagonal with float32
    input_dict = {
        "subscripts": "ii...->...",
        "operands": np.random.randn(3, 3, 4).astype(np.float32),
        "optimize": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.einsum_path_1"] = jax_numpy_einsum_path_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_path_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_path_1'.")


check_valid('jax.numpy.einsum_path', generated_inputs['jax.numpy.einsum_path_1'], lib="jax", suffix=1)
