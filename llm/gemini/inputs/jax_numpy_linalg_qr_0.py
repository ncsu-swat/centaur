
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def qr_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, reduced mode
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"a": a, "mode": "reduced"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall matrix, complete mode
    a = np.random.randn(6, 4).astype(np.float32)
    input_dict = {"a": a, "mode": "complete"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide matrix, 'r' mode, float64
    a = np.random.randn(3, 7).astype(np.float64)
    input_dict = {"a": a, "mode": "r"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex matrix, raw mode
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {"a": a, "mode": "raw"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched square matrix, reduced mode
    a = np.random.randn(2, 5, 5).astype(np.float32)
    input_dict = {"a": a, "mode": "reduced"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-batch tall matrix, complete mode, float64
    a = np.random.randn(3, 2, 6, 4).astype(np.float64)
    input_dict = {"a": a, "mode": "complete"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small matrix with negative values, 'r' mode
    a = np.array([[-1.0, 2.0], [-3.0, -4.0], [5.0, -6.0]], dtype=np.float32)
    input_dict = {"a": a, "mode": "r"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 matrix, reduced mode
    a = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex128)
    input_dict = {"a": a, "mode": "reduced"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large square matrix, raw mode
    a = np.random.randn(16, 16).astype(np.float32)
    input_dict = {"a": a, "mode": "raw"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched tall matrix, complete mode
    a = np.random.randn(4, 8, 5).astype(np.float32)
    input_dict = {"a": a, "mode": "complete"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.qr"] = qr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.qr'.")


check_valid('jax.numpy.linalg.qr', generated_inputs['jax.numpy.linalg.qr'], lib="jax", suffix=0)
