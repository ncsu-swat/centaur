
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Trace of a 2D square matrix
    input_dict = {
        "subscripts": "ii->",
        "operands": np.random.randn(4, 4).astype(np.float32),
        "optimize": "auto",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Diagonal of a 2D square matrix
    input_dict = {
        "subscripts": "ii->i",
        "operands": np.random.randn(5, 5).astype(np.float32),
        "optimize": "greedy",
        "precision": "HIGH",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transpose of a 2D matrix
    input_dict = {
        "subscripts": "ij->ji",
        "operands": np.random.randn(3, 5).astype(np.float64),
        "optimize": "optimal",
        "precision": "HIGHEST",
        "preferred_element_type": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sum of a 1D vector
    input_dict = {
        "subscripts": "i->",
        "operands": np.random.randn(10).astype(np.float32),
        "optimize": "auto",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sum along columns of a 2D matrix
    input_dict = {
        "subscripts": "ij->j",
        "operands": np.random.randn(4, 6).astype(np.float32),
        "optimize": "eager",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sum along rows of a 2D matrix
    input_dict = {
        "subscripts": "ij->i",
        "operands": np.random.randint(-10, 10, size=(5, 3)).astype(np.int32),
        "optimize": "auto",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Transpose of a 3D tensor
    input_dict = {
        "subscripts": "ijk->kij",
        "operands": np.random.randn(2, 3, 4).astype(np.float32),
        "optimize": "optimal",
        "precision": "HIGH",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Sum all elements of a 2D matrix
    input_dict = {
        "subscripts": "ij->",
        "operands": np.random.randn(4, 4).astype(np.float32),
        "optimize": "greedy",
        "precision": "HIGHEST",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Permute axes of a 4D tensor
    input_dict = {
        "subscripts": "ijkl->ljki",
        "operands": np.random.randn(2, 2, 3, 3).astype(np.float32),
        "optimize": "auto",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sum one dimension of a 3D tensor
    input_dict = {
        "subscripts": "ijk->ik",
        "operands": np.random.randn(2, 4, 3).astype(np.float32),
        "optimize": "auto",
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.einsum_1"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_1'.")


check_valid('jax.numpy.einsum', generated_inputs['jax.numpy.einsum_1'], lib="jax", suffix=1)
