
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"x": x, "keepdims": True, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 4, 4)).astype(np.float32)
    input_dict = {"x": x, "keepdims": True, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"x": x, "keepdims": False, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = (np.ones((10, 10)) * -2.5).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.randn(10, 2).astype(np.float32)
    input_dict = {"x": x, "keepdims": True, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.randn(5, 5, 2, 2).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.randn(50, 50).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matrix_norm_1"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_norm_1'.")


check_valid('jax.numpy.linalg.matrix_norm', generated_inputs['jax.numpy.linalg.matrix_norm_1'], lib="jax", suffix=1)
