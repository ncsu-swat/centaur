
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matvec_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D matrix and 1D vector (float32)
    x1 = np.random.randn(4, 5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex values
    x1 = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    x2 = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer values (with negatives)
    x1 = np.random.randint(-10, 10, size=(2, 3)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(3,)).astype(np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched matrix-vector product (same leading dimensions)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 4).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched with broadcasting (x2 is 1D vector broadcasted)
    x1 = np.random.randn(3, 5, 2).astype(np.float64)
    x2 = np.random.randn(2).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched with broadcasting (x1 has leading dim 1, x2 has leading dim 3)
    x1 = np.random.randn(1, 4, 3).astype(np.float32)
    x2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large dimensions
    x1 = np.random.randn(64, 128).astype(np.float32)
    x2 = np.random.randn(128).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Higher dimensional batching
    x1 = np.random.randint(-5, 5, size=(2, 3, 4, 5)).astype(np.int64)
    x2 = np.random.randint(-5, 5, size=(2, 3, 5)).astype(np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean types
    x1 = np.random.choice([True, False], size=(2, 2))
    x2 = np.random.choice([True, False], size=(2,))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 precision
    x1 = np.random.randn(5, 5).astype(np.float64)
    x2 = np.random.randn(5).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative float values
    x1 = -np.abs(np.random.randn(3, 3).astype(np.float32))
    x2 = -np.abs(np.random.randn(3).astype(np.float32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.matvec"] = matvec_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matvec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matvec'.")


check_valid('jax.numpy.matvec', generated_inputs['jax.numpy.matvec'], lib="jax", suffix=0)
