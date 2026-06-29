
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, positive x and beta
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    beta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with negative values for x, float64
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    beta = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar beta (0D array), 2D array for x
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    beta = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays for both, same shape
    x = np.random.randn(3, 3).astype(np.float32)
    beta = np.random.uniform(0.1, 5.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting, 2D x and 1D beta
    x = np.random.randn(2, 4).astype(np.float64)
    beta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large dimensions (3D arrays), float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    beta = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small beta values (approaching laplace-like)
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    beta = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large beta values (approaching uniform-like)
    x = np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    beta = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element 1D arrays
    x = np.array([0.0], dtype=np.float64)
    beta = np.array([2.0], dtype=np.float64)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting x (1D) with beta (2D)
    x = np.array([-1.0, 1.0], dtype=np.float32)
    beta = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.cdf_1"] = gennorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.cdf_1'.")


check_valid('jax.scipy.stats.gennorm.cdf', generated_inputs['jax.scipy.stats.gennorm.cdf_1'], lib="jax", suffix=1)
