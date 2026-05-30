
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def einsum_path_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 matrix, diagonal sum
    subscripts = "ii"
    operands = np.random.randn(5, 5).astype(np.float32)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 2: 3D float64 tensor, contracting two dimensions
    subscripts = "iij"
    operands = np.random.randn(4, 4, 3).astype(np.float64)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 3: 4D int32 tensor, contracting all dimensions to scalar
    subscripts = "iijj"
    operands = np.random.randint(-5, 5, size=(2, 2, 3, 3)).astype(np.int32)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 4: 1D int64 array, trivial sum
    subscripts = "i"
    operands = np.random.randint(-100, 100, size=(10,)).astype(np.int64)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 5: 2D complex64 matrix, transpose-like contraction
    subscripts = "ij->ji"
    operands = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 6: 5D float32 tensor, contracting some dimensions
    subscripts = "abcad->bcd"
    operands = np.random.randn(2, 3, 4, 2, 5).astype(np.float32)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 7: 3D float32 tensor, reducing to 1D
    subscripts = "aba->b"
    operands = np.random.randn(5, 10, 5).astype(np.float32)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 8: 2D int32 matrix, trace with negative values
    subscripts = "ii->i"
    operands = np.random.randint(-10, 10, size=(6, 6)).astype(np.int32)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 9: 4D complex128 tensor
    subscripts = "abab->ab"
    real_part = np.random.randn(2, 3, 2, 3)
    imag_part = np.random.randn(2, 3, 2, 3)
    operands = (real_part + 1j * imag_part).astype(np.complex128)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    # Input 10: 3D float64 tensor, trivial transpose
    subscripts = "ijk->kij"
    operands = np.random.randn(2, 3, 4).astype(np.float64)
    optimize = [(0,)]
    list_of_inputs.append({
        "subscripts": subscripts,
        "operands": operands,
        "optimize": optimize
    })

    return list_of_inputs

generated_inputs["jax.numpy.einsum_path_3"] = einsum_path_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_path_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_path_3'.")


check_valid('jax.numpy.einsum_path', generated_inputs['jax.numpy.einsum_path_3'], lib="jax", suffix=3)
