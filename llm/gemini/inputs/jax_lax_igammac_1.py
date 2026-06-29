
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def igammac_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays, same shape (3,)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, same shape (3, 3)
    a = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    x = np.random.uniform(0.0, 10.0, size=(3, 3)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 precision, same shape (4, 4)
    a = np.random.uniform(0.1, 100.0, size=(4, 4)).astype(np.float64)
    x = np.random.uniform(0.0, 100.0, size=(4, 4)).astype(np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar-like arrays, same shape ()
    a = np.array(5.0, dtype=np.float32)
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D arrays, same shape (100,)
    a = np.random.uniform(1.0, 50.0, size=(100,)).astype(np.float32)
    x = np.random.uniform(0.0, 50.0, size=(100,)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, same shape (2, 3, 4)
    a = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    x = np.random.uniform(0.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D arrays, same shape (5, 5)
    a = np.random.uniform(0.1, 10.0, size=(5, 5)).astype(np.float32)
    x = np.random.uniform(0.0, 10.0, size=(5, 5)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays, same shape (1,)
    a = np.random.uniform(1.0, 10.0, size=(1,)).astype(np.float64)
    x = np.random.uniform(0.0, 10.0, size=(1,)).astype(np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays, same shape (2, 2, 2, 2)
    a = np.random.uniform(0.1, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    x = np.random.uniform(0.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small positive values, same shape (5,)
    a = np.random.uniform(1e-5, 1e-2, size=(5,)).astype(np.float32)
    x = np.random.uniform(0.0, 1e-2, size=(5,)).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High values, same shape (2, 2)
    a = np.random.uniform(100.0, 1000.0, size=(2, 2)).astype(np.float64)
    x = np.random.uniform(100.0, 1000.0, size=(2, 2)).astype(np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.igammac_1"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igammac_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igammac_1'.")


check_valid('jax.lax.igammac', generated_inputs['jax.lax.igammac_1'], lib="jax", suffix=1)
