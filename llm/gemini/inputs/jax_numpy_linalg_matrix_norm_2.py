
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_matrix_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix, ord=1, keepdims=False
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrix with float64, ord=2, keepdims=True
    x = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Stack of matrices (3D), ord=-1, keepdims=False
    x = np.random.randn(2, 5, 5).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stack of matrices, ord=-2, keepdims=True
    x = np.random.randn(3, 2, 4).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix with negative values, ord=1, keepdims=False
    x = (np.random.randn(5, 3) * 10).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor representing a batch of matrices, ord=2, keepdims=True
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 2D matrix, ord=-1, keepdims=False
    x = np.random.randn(10, 10).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 2D matrix, ord=-2, keepdims=True
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rectangular matrix, ord=1, keepdims=True
    x = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"x": x, "keepdims": True, "ord": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Stack of rectangular matrices, ord=2, keepdims=False
    x = np.random.randn(2, 3, 5).astype(np.float64)
    input_dict = {"x": x, "keepdims": False, "ord": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matrix_norm_2"] = jax_numpy_linalg_matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_norm_2'.")


check_valid('jax.numpy.linalg.matrix_norm', generated_inputs['jax.numpy.linalg.matrix_norm_2'], lib="jax", suffix=2)
