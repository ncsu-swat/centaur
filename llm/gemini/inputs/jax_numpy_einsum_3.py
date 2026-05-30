
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class PretendList(np.ndarray):
    @property
    def __class__(self):
        return list

def make_pretend_list(arr):
    return arr.view(PretendList)

def einsum_inputs():
    list_of_inputs = []

    # 1. Trace of 2D matrix
    input_dict = {
        "subscripts": "ii",
        "operands": make_pretend_list(np.random.randn(5, 5).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Matrix Transpose
    input_dict = {
        "subscripts": "ij->ji",
        "operands": make_pretend_list(np.random.randn(4, 7).astype(np.float64)),
        "optimize": [],
        "precision": "HIGHEST",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Sum of 1D array
    input_dict = {
        "subscripts": "i->",
        "operands": make_pretend_list(np.random.randn(10).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Sum of 2D matrix
    input_dict = {
        "subscripts": "ij->",
        "operands": make_pretend_list(np.random.randn(6, 6).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Sum along columns of 2D matrix
    input_dict = {
        "subscripts": "ij->i",
        "operands": make_pretend_list(np.random.randn(5, 8).astype(np.float32)),
        "optimize": [],
        "precision": "HIGH",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. Sum along rows of 2D matrix
    input_dict = {
        "subscripts": "ij->j",
        "operands": make_pretend_list(np.random.randn(5, 8).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Matrix Diagonal
    input_dict = {
        "subscripts": "ii->i",
        "operands": make_pretend_list(np.random.randn(6, 6).astype(np.float32)),
        "optimize": [],
        "precision": "HIGH",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Sum along axis 2 of 3D tensor
    input_dict = {
        "subscripts": "ijk->ij",
        "operands": make_pretend_list(np.random.randn(2, 3, 4).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Permute axes of 3D tensor
    input_dict = {
        "subscripts": "ijk->kij",
        "operands": make_pretend_list(np.random.randn(2, 3, 4).astype(np.float32)),
        "optimize": [],
        "precision": "DEFAULT",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. Sum with ellipsis
    input_dict = {
        "subscripts": "...j->...",
        "operands": make_pretend_list(np.random.randn(2, 3, 4).astype(np.float32)),
        "optimize": [],
        "precision": "HIGH",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.einsum_3"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_3'.")


check_valid('jax.numpy.einsum', generated_inputs['jax.numpy.einsum_3'], lib="jax", suffix=3)
