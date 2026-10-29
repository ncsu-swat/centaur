
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_inputs():
    list_of_inputs = []

    # Input 1: Trace of a square matrix
    subscripts = "ii->"
    operands = np.arange(16, dtype=np.int32).reshape(4, 4) - 5
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "auto",
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix transpose
    subscripts = "ij->ji"
    operands = np.random.randn(3, 7).astype(np.float32)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "optimal",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Diagonal extraction
    subscripts = "ii->i"
    operands = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "greedy",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Sum along axis
    subscripts = "...j->..."
    operands = np.random.randn(4, 3, 2).astype(np.float32)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "eager",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sum of 1D vector
    subscripts = "i->"
    operands = np.array([-1, 0, 2, 4, 8], dtype=np.int64)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "auto",
        "precision": "default",
        "preferred_element_type": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher-dimensional transpose/permutation
    subscripts = "ijkl->ljik"
    operands = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "optimal",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Partial trace (contracting two dimensions of 3D tensor)
    subscripts = "ijj->i"
    operands = np.random.randn(3, 4, 4).astype(np.float64)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "greedy",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch trace with ellipsis
    subscripts = "...ii->..."
    operands = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "auto",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identity-like copy / view
    subscripts = "ij->ij"
    operands = np.random.randn(6, 6).astype(np.float32)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "auto",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-axis reduction to scalar
    subscripts = "ijk->"
    operands = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "subscripts": subscripts,
        "operands": operands,
        "optimize": "optimal",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
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
