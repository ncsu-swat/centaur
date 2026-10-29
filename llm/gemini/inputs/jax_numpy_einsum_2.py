
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix diagonal extraction (2D square matrix)
    m1 = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ii->i",
                "operands": m1,
                "optimize": True,
                "precision": (0, 0),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 2: Matrix trace reduction (2D square matrix with negative values)
    m2 = np.array(
        [[-2.5, 1.0, 3.5], [0.5, -4.0, 2.0], [1.5, -0.5, 6.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ii->",
                "operands": m2,
                "optimize": False,
                "precision": (1, 1),
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 3: Matrix transpose (2D rectangular matrix)
    m3 = np.random.randn(4, 7).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ij->ji",
                "operands": m3,
                "optimize": True,
                "precision": (2, 2),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 4: 1D array total sum reduction (integer dtype)
    v1 = np.arange(-10, 10, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "i->",
                "operands": v1,
                "optimize": False,
                "precision": (0, 0),
                "preferred_element_type": np.dtype(np.int32),
            }
        )
    )

    # Input 5: Sum along axis (2D matrix row-sum)
    m4 = np.random.randn(6, 8).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ij->i",
                "operands": m4,
                "optimize": True,
                "precision": (1, 1),
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 6: 3D tensor axis permutation
    t1 = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ijk->kji",
                "operands": t1,
                "optimize": False,
                "precision": (0, 0),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 7: Batched diagonal extraction (3D tensor)
    t2 = np.random.randn(4, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "bij->bi",
                "operands": t2,
                "optimize": True,
                "precision": (2, 2),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 8: Batched trace reduction (3D tensor)
    t3 = np.random.randn(8, 4, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "bii->b",
                "operands": t3,
                "optimize": False,
                "precision": (1, 1),
                "preferred_element_type": np.dtype(np.float64),
            }
        )
    )

    # Input 9: 4D tensor dimension reversal
    t4 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "abcd->dcba",
                "operands": t4,
                "optimize": True,
                "precision": (0, 0),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 10: Diagonal contraction of last two axes in 3D tensor
    t5 = np.random.randn(4, 5, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ijj->i",
                "operands": t5,
                "optimize": False,
                "precision": (1, 1),
                "preferred_element_type": np.dtype(np.float32),
            }
        )
    )

    # Input 11: Complex-valued matrix diagonal
    c = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "subscripts": "ii->i",
                "operands": c,
                "optimize": True,
                "precision": (0, 0),
                "preferred_element_type": np.dtype(np.complex64),
            }
        )
    )

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
