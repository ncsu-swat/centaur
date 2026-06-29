
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matrix_transpose_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D float32 matrix
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 matrix with negative/positive elements
    x = np.random.uniform(-10.0, 10.0, size=(5, 5)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array (stack of matrices)
    x = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 4D int64 array
    x = np.random.randint(0, 10, size=(2, 2, 4, 3)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D complex64 matrix
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D boolean array
    x = np.random.choice([True, False], size=(3, 2, 2)).astype(bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 5D float32 array (deep stack of matrices)
    x = np.random.randn(2, 2, 2, 5, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D float32 matrix with a single row
    x = np.random.randn(1, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D float32 matrix with a single column
    x = np.random.randn(5, 1).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Large 2D float32 matrix
    x = np.random.randn(100, 200).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matrix_transpose"] = matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_transpose'.")


check_valid('jax.numpy.linalg.matrix_transpose', generated_inputs['jax.numpy.linalg.matrix_transpose'], lib="jax", suffix=0)
