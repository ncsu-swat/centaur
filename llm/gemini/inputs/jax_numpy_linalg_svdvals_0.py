
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def svdvals_inputs():
    list_of_inputs = []

    # Input 1: Square matrix (float32)
    x = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Wide matrix (float32)
    x = np.random.randn(4, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Tall matrix (float32)
    x = np.random.randn(8, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Matrix with negative values (float64)
    x = (np.random.randn(6, 6) - 5.0).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Float64 precision square matrix
    x = np.random.randn(10, 10).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D batched matrices (shape: 2, 4, 4)
    x = np.random.randn(2, 4, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D batched matrices (shape: 2, 3, 5, 3)
    x = np.random.randn(2, 3, 5, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Complex64 matrix
    real = np.random.randn(4, 4).astype(np.float32)
    imag = np.random.randn(4, 4).astype(np.float32)
    x = real + 1j * imag
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Complex128 batched matrix
    real = np.random.randn(2, 3, 3).astype(np.float64)
    imag = np.random.randn(2, 3, 3).astype(np.float64)
    x = real + 1j * imag
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Identity matrix (float32)
    x = np.eye(6, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Matrix with large scale values
    x = (np.random.randn(5, 7) * 100.0).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.svdvals"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.svdvals'.")


check_valid('jax.numpy.linalg.svdvals', generated_inputs['jax.numpy.linalg.svdvals'], lib="jax", suffix=0)
