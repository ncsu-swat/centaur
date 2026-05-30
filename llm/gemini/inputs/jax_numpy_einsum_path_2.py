
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def einsum_path_inputs():
    list_of_inputs = []

    # Input 1: Trace of a 2D matrix
    input_dict = {
        "subscripts": "ii",
        "operands": np.random.randn(3, 3).astype(np.float32),
        "optimize": "optimal"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Transpose of a 2D matrix
    input_dict = {
        "subscripts": "ij->ji",
        "operands": np.random.randn(4, 5).astype(np.float32),
        "optimize": "greedy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sum of 3D tensor over last axis
    input_dict = {
        "subscripts": "ijk->ij",
        "operands": np.random.randn(2, 3, 4).astype(np.float64),
        "optimize": "eager"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Diagonal extraction from 4D tensor
    input_dict = {
        "subscripts": "iijk->ijk",
        "operands": np.random.randint(-10, 10, size=(2, 2, 3, 4)).astype(np.int32),
        "optimize": "auto"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Vector reduction
    input_dict = {
        "subscripts": "i->",
        "operands": np.random.randn(10).astype(np.float32),
        "optimize": "optimal"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Reduction of 3D tensor with shared axes
    input_dict = {
        "subscripts": "iji->j",
        "operands": np.random.randn(3, 4, 3).astype(np.float64),
        "optimize": "greedy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large matrix transpose
    input_dict = {
        "subscripts": "ij->ji",
        "operands": np.random.randn(100, 200).astype(np.float32),
        "optimize": "auto"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D tensor reduction
    input_dict = {
        "subscripts": "abcde->bd",
        "operands": np.random.randn(2, 3, 4, 5, 6).astype(np.float32),
        "optimize": "eager"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Trace with negative float64 values
    input_dict = {
        "subscripts": "ii->",
        "operands": np.random.uniform(-100.0, -1.0, size=(5, 5)).astype(np.float64),
        "optimize": "optimal"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple 2D sum to scalar
    input_dict = {
        "subscripts": "ij->",
        "operands": np.random.randint(-100, 100, size=(10, 10)).astype(np.int64),
        "optimize": "greedy"
    }
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
