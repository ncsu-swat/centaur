
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cond_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 float32 matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 float64 matrix with random values
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 complex64 matrix
    x = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 5x5 complex128 matrix
    x = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex128)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D tensor (2, 3, 3) float32
    x = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher-dimensional batched tensor (3, 2, 4, 4) float64
    x = np.random.randn(3, 2, 4, 4).astype(np.float64)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2x2 matrix with negative values
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Well-conditioned diagonal-like 6x6 matrix
    x = (np.eye(6) * 5.0).astype(np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Ill-conditioned/singular 2x2 matrix
    x = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x5x5 float32 matrix
    x = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict = {"x": x, "p": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.cond_3"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cond_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cond_3'.")


check_valid('jax.numpy.linalg.cond', generated_inputs['jax.numpy.linalg.cond_3'], lib="jax", suffix=3)
