
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D array, beta = 2.0 (standard normal equivalent)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    beta = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 2: 2D array, beta = 1.0 (Laplace distribution equivalent)
    x = np.array([[-1.5, 0.5], [2.3, -0.7]], dtype=np.float64)
    beta = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 3: 0D scalar array, beta = 0.5 (heavy-tailed)
    x = np.array(0.0, dtype=np.float32)
    beta = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 4: 3D array of zeros and ones, beta = 4.0
    x = np.ones((2, 3, 4), dtype=np.float32)
    beta = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 5: 1D array with large positive and negative values, beta = 3.0
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    beta = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 6: 2D array with random normal values, beta = 1.5
    x = np.random.randn(4, 5).astype(np.float32)
    beta = 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 7: 4D tensor with small values, beta = 0.8
    x = np.linspace(-0.5, 0.5, 16).reshape(2, 2, 2, 2).astype(np.float32)
    beta = 0.8
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 8: 1D array float64, beta = 5.5
    x = np.linspace(-3.0, 3.0, 50, dtype=np.float64)
    beta = 5.5
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 9: 2D array with strictly negative values, beta = 2.5
    x = -np.abs(np.random.randn(3, 3)).astype(np.float32) - 0.1
    beta = 2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 10: 1D single element array, beta = 10.0
    x = np.array([1.23], dtype=np.float64)
    beta = 10.0
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 11: 3D array with extreme values, beta = 0.1
    x = np.array([[[-1e-3, 1e-3]], [[-2.0, 2.0]]], dtype=np.float32)
    beta = 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.logpdf_2"] = gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_2'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_2'], lib="jax", suffix=2)
