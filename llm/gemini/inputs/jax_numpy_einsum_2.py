
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=dtype or np.int32)

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix trace
    list_of_inputs.append({
        "subscripts": "ii",
        "operands": np.random.randn(5, 5).astype(np.float32),
        "optimize": True,
        "precision": SafeTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 2: Matrix diagonal
    list_of_inputs.append({
        "subscripts": "ii->i",
        "operands": np.random.randn(4, 4).astype(np.float32),
        "optimize": False,
        "precision": SafeTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 3: Matrix transpose
    list_of_inputs.append({
        "subscripts": "ij->ji",
        "operands": np.random.randn(3, 5).astype(np.float64),
        "optimize": True,
        "precision": SafeTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.float64)
    })

    # Input 4: Sum of all elements
    list_of_inputs.append({
        "subscripts": "ij->",
        "operands": np.random.randn(4, 3).astype(np.float32),
        "optimize": False,
        "precision": SafeTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 5: Sum along axis 1
    list_of_inputs.append({
        "subscripts": "ij->i",
        "operands": np.random.randn(5, 3).astype(np.float32),
        "optimize": True,
        "precision": SafeTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 6: Sum along axis 0
    list_of_inputs.append({
        "subscripts": "ij->j",
        "operands": np.random.randn(2, 4).astype(np.float32),
        "optimize": False,
        "precision": SafeTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 7: 3D Transpose
    list_of_inputs.append({
        "subscripts": "ijk->kij",
        "operands": np.random.randn(2, 3, 4).astype(np.float32),
        "optimize": True,
        "precision": SafeTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 8: 3D Sum along axis 1
    list_of_inputs.append({
        "subscripts": "ijk->ik",
        "operands": np.random.randn(2, 3, 4).astype(np.float64),
        "optimize": True,
        "precision": SafeTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float64)
    })

    # Input 9: 3D Partial trace
    list_of_inputs.append({
        "subscripts": "iij->j",
        "operands": np.random.randn(3, 3, 4).astype(np.float32),
        "optimize": True,
        "precision": SafeTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 10: 4D reduction
    list_of_inputs.append({
        "subscripts": "ijkl->ki",
        "operands": np.random.randn(2, 3, 2, 4).astype(np.float32),
        "optimize": False,
        "precision": SafeTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_2'.")


check_valid('jax.numpy.einsum', generated_inputs['jax.numpy.einsum_2'], lib="jax", suffix=2)
