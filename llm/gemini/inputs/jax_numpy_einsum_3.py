
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_inputs():
    list_of_inputs = []

    # Input 1: 1D array sum
    x = np.random.randn(8).astype(np.float32)
    input_dict = {
        "subscripts": "i->",
        "operands": x,
        "optimize": [(0,)],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix transpose (2D, float64)
    m = np.random.randn(4, 6).astype(np.float64)
    input_dict = {
        "subscripts": "ij->ji",
        "operands": m,
        "optimize": [(0,)],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix diagonal extraction
    m = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "subscripts": "ii->i",
        "operands": m,
        "optimize": [(0,)],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix trace (reduction to scalar)
    m = np.random.randn(6, 6).astype(np.float32)
    input_dict = {
        "subscripts": "ii->",
        "operands": m,
        "optimize": [(0,)],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sum along an axis using ellipsis (3D tensor)
    t = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "subscripts": "...j->...",
        "operands": t,
        "optimize": [(0,)],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Full reduction of a 2D matrix
    m = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "subscripts": "ij->",
        "operands": m,
        "optimize": [(0,)],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor permutation / transpose
    t = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "subscripts": "ijk->kji",
        "operands": t,
        "optimize": [(0,)],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Partial reduction of a 3D tensor along one axis
    t = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "subscripts": "ijk->ik",
        "operands": t,
        "optimize": [(0,)],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor axes permutation
    t = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "subscripts": "ijkl->lkji",
        "operands": t,
        "optimize": [(0,)],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D identity / copy
    x = np.random.randn(12).astype(np.float32)
    input_dict = {
        "subscripts": "i->i",
        "operands": x,
        "optimize": [(0,)],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Integer matrix trace with negative values
    m = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    input_dict = {
        "subscripts": "ii->",
        "operands": m,
        "optimize": [(0,)],
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
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
