
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matrix_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 square matrix
    x = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 rectangular matrix
    x = np.random.randn(4, 7).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 batched square matrix
    x = np.random.randint(-10, 10, size=(2, 5, 5)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 batched rectangular matrix with negative values
    x = np.random.uniform(-5.0, 5.0, size=(3, 2, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float64 multi-batched matrix
    x = np.random.randn(2, 2, 3, 6).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D complex64 matrix
    x = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D boolean matrix
    x = np.random.choice([True, False], size=(4, 4))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 5D int64 matrix with larger batch dimensions
    x = np.random.randint(0, 100, size=(2, 1, 3, 4, 2)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D float32 matrix with extremely large numbers
    x = (np.random.randn(5, 5) * 1e5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 3D float32 matrix where one of the last two dimensions is 1
    x = np.random.randn(4, 1, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.matrix_transpose"] = matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matrix_transpose'.")


check_valid('jax.numpy.matrix_transpose', generated_inputs['jax.numpy.matrix_transpose'], lib="jax", suffix=0)
