
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32
    x = np.array(0.5, dtype=np.float32)
    beta = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 2: 1D arrays with standard normal equivalent beta=2.0
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    beta = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 3: 1D array with Laplace equivalent beta=1.0, float64
    x = np.array([-3.5, -0.5, 0.0, 1.2, 4.8], dtype=np.float64)
    beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 4: 2D arrays, same shape
    x = np.array([[-1.5, 0.2], [1.1, -0.8]], dtype=np.float32)
    beta = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 5: Broadcasting 2D with 1D
    x = np.random.randn(3, 4).astype(np.float32)
    beta = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 6: Broadcasting (4, 1) and (1, 3)
    x = np.array([[-2.0], [-0.5], [0.5], [2.0]], dtype=np.float64)
    beta = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 7: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    beta = np.full((2, 3, 4), 1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 8: Extreme tail values
    x = np.array([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=np.float32)
    beta = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 9: Small beta values (heavy-tailed)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    beta = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 10: Large beta values (approaching uniform)
    x = np.linspace(-2.0, 2.0, 6, dtype=np.float64)
    beta = np.full(6, 10.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 11: 4D tensor inputs
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    beta = (np.random.rand(2, 2, 2, 2) + 0.5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

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
