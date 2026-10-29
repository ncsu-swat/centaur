
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D beta, positive x
    x = 1.5
    beta = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 2: 2D beta, negative x
    x = -2.0
    beta = np.array([[1.0, 2.5], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 3: 0D tensor (scalar array) beta, zero x
    x = 0.0
    beta = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 4: 3D tensor beta with varying shape parameters, positive x
    x = 0.5
    beta = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 5: Large positive x, small beta values (heavy-tailed)
    x = 10.0
    beta = np.array([0.1, 0.2, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 6: Large negative x, large beta values (thin-tailed)
    x = -5.0
    beta = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 7: 4D beta tensor
    x = -0.75
    beta = np.full((2, 2, 2, 2), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 8: 1D beta with int-compatible float values, float64
    x = 3.14159
    beta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 9: Small fractional x near zero
    x = 1e-4
    beta = np.linspace(0.5, 5.0, 10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 10: 2D beta tensor with non-square shape
    x = -1.23
    beta = np.array(
        [[0.8, 1.2, 1.6, 2.0], [2.4, 2.8, 3.2, 3.6]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.logpdf_4"] = gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_4'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_4'], lib="jax", suffix=4)
