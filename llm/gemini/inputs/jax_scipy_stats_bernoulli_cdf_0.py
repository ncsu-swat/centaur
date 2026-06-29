
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bernoulli_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D arrays (scalars as tensors)
    k = np.array(0, dtype=np.int32)
    p = np.array(0.5, dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of the same size, basic float values
    k = np.array([-1, 0, 1, 2], dtype=np.float32)
    p = np.array([0.1, 0.5, 0.9, 0.5], dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcast a scalar 'p' over 1D 'k'
    k = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    p = np.array(0.3, dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcast a scalar 'k' over 1D 'p'
    k = np.array(0, dtype=np.int32)
    p = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 arrays, 2D shape
    k = np.array([[0, 1], [1, 0]], dtype=np.float64)
    p = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float64)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, with mixed float values
    k = np.random.choice([-1.0, 0.0, 1.0, 2.0], size=(2, 2, 2)).astype(np.float32)
    p = np.random.uniform(0.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean 'k' array and float 'p' array
    k = np.array([True, False, True], dtype=bool)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting higher dimensions
    k = np.array([[0], [1]], dtype=np.int32)
    p = np.array([[0.2, 0.5, 0.8]], dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array
    k = np.linspace(-1, 2, 100).astype(np.float32)
    p = np.array(0.5, dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boundary values for probability
    k = np.array([0, 1, 0, 1], dtype=np.float32)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"k": k, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.cdf"] = bernoulli_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.cdf'.")


check_valid('jax.scipy.stats.bernoulli.cdf', generated_inputs['jax.scipy.stats.bernoulli.cdf'], lib="jax", suffix=0)
