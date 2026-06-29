
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def geom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with integer k and float32 p, loc=0
    k = np.array([1, 2, 3, 4], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar arrays
    k = np.array(5, dtype=np.int32)
    p = np.array(0.1, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with float32 types
    k = np.array([[2, 3], [4, 5]], dtype=np.float32)
    p = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    loc = np.array([[0, 0], [0, 0]], dtype=np.float32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting (k is 2D column, p is 2D row, loc is scalar)
    k = np.array([[1], [2], [3]], dtype=np.int32)
    p = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    loc = np.array([[0]], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-zero location offset
    k = np.array([3, 4, 5], dtype=np.int32)
    p = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    loc = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 precision arrays
    k = np.array([10, 20], dtype=np.float64)
    p = np.array([0.05, 0.01], dtype=np.float64)
    loc = np.array([2, 5], dtype=np.float64)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays
    k = np.ones((2, 2, 2), dtype=np.int32) * 3
    p = np.full((2, 2, 2), 0.75, dtype=np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Edge case with p close to 1
    k = np.array([100], dtype=np.int32)
    p = np.array([0.99], dtype=np.float32)
    loc = np.array([10], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Edge case with very small p
    k = np.array([5, 10], dtype=np.int32)
    p = np.array([1e-5, 1e-4], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixing integer and float types for loc and k
    k = np.array([2.0, 4.0], dtype=np.float32)
    p = np.array([0.3, 0.7], dtype=np.float32)
    loc = np.array([1, 2], dtype=np.int32)
    input_dict = {"k": k, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.geom.logpmf"] = geom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.geom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.geom.logpmf'.")


check_valid('jax.scipy.stats.geom.logpmf', generated_inputs['jax.scipy.stats.geom.logpmf'], lib="jax", suffix=0)
