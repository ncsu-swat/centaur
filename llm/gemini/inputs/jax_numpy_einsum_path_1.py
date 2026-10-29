
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_path_inputs():
    list_of_inputs = []

    # Input 1: 2D transpose (float32)
    operands = np.random.randn(5, 7).astype(np.float32)
    list_of_inputs.append(
        {"subscripts": "ij->ji", "operands": operands, "optimize": True}
    )

    # Input 2: 2D trace (int32, negative values)
    operands = np.random.randint(-10, 10, size=(4, 4), dtype=np.int32)
    list_of_inputs.append(
        {"subscripts": "ii", "operands": operands, "optimize": False}
    )

    # Input 3: 1D sum reduction (float64)
    operands = np.random.randn(100).astype(np.float64)
    list_of_inputs.append(
        {"subscripts": "i->", "operands": operands, "optimize": True}
    )

    # Input 4: 2D row sum reduction (float32)
    operands = np.random.randn(8, 12).astype(np.float32)
    list_of_inputs.append(
        {"subscripts": "ij->i", "operands": operands, "optimize": False}
    )

    # Input 5: 3D tensor partial trace (complex64)
    operands = (
        np.random.randn(6, 4, 6) + 1j * np.random.randn(6, 4, 6)
    ).astype(np.complex64)
    list_of_inputs.append(
        {"subscripts": "iji->j", "operands": operands, "optimize": True}
    )

    # Input 6: 4D tensor axis permutation (int64)
    operands = np.random.randint(-5, 5, size=(2, 3, 4, 5), dtype=np.int64)
    list_of_inputs.append(
        {"subscripts": "ijkl->lkji", "operands": operands, "optimize": True}
    )

    # Input 7: 2D diagonal extraction (float32)
    operands = np.random.randn(6, 6).astype(np.float32)
    list_of_inputs.append(
        {"subscripts": "ii->i", "operands": operands, "optimize": False}
    )

    # Input 8: 3D tensor total reduction (float64)
    operands = np.random.randn(3, 4, 5).astype(np.float64)
    list_of_inputs.append(
        {"subscripts": "ijk->", "operands": operands, "optimize": True}
    )

    # Input 9: 5D tensor permutation (float16)
    operands = np.random.randn(2, 2, 3, 2, 4).astype(np.float16)
    list_of_inputs.append(
        {"subscripts": "abcde->edcba", "operands": operands, "optimize": True}
    )

    # Input 10: 1D identity copy (int8)
    operands = np.random.randint(-128, 127, size=(16,), dtype=np.int8)
    list_of_inputs.append(
        {"subscripts": "i->i", "operands": operands, "optimize": False}
    )

    # Input 11: 3D slice reduction to matrix (float32)
    operands = np.random.randn(4, 5, 6).astype(np.float32)
    list_of_inputs.append(
        {"subscripts": "ijk->ik", "operands": operands, "optimize": True}
    )

    return list_of_inputs


generated_inputs["jax.numpy.einsum_path_1"] = einsum_path_inputs()

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
