
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bernoulli_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays with default float32/int32
    k = np.array([0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalar representation)
    k = np.array(1, dtype=np.int32)
    p = np.array(0.3, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float type for 'k'
    k = np.array([0.0, 1.0], dtype=np.float32)
    p = np.array([0.2, 0.8], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    k = np.array([[0, 1], [1, 0]], dtype=np.int32)
    p = np.array([[0.4, 0.6], [0.1, 0.9]], dtype=np.float32)
    loc = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean 'k' array
    k = np.array([True, False, True], dtype=bool)
    p = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes
    k = np.array([0, 1, 2], dtype=np.int32)
    p = np.array([[0.5], [0.3]], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Non-zero location (shift)
    k = np.array([1, 2, 3], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 precision
    k = np.array([0, 1], dtype=np.int64)
    p = np.array([0.123456789, 0.987654321], dtype=np.float64)
    loc = np.array([0, 0], dtype=np.int64)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Out of bound values for 'k'
    k = np.array([-1, 2, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Probability boundary cases
    k = np.array([0, 1], dtype=np.int32)
    p = np.array([1e-7, 1.0 - 1e-7], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.logpmf"] = bernoulli_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.logpmf'.")


check_valid('jax.scipy.stats.bernoulli.logpmf', generated_inputs['jax.scipy.stats.bernoulli.logpmf'], lib="jax", suffix=0)
