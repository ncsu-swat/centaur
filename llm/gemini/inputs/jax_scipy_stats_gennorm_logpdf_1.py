
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard normal case (beta=2.0)
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    beta = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 2: 0D scalar arrays, Laplace case (beta=1.0)
    x = np.array(1.5, dtype=np.float64)
    beta = np.array(1.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 3: 2D arrays, float64
    x = np.array([[-1.2, 0.3], [2.1, -0.5]], dtype=np.float64)
    beta = np.array([[0.5, 1.5], [2.5, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 4: Broadcasting (x 2D, beta 1D)
    x = np.random.randn(3, 4).astype(np.float32)
    beta = np.array([0.8, 1.2, 2.0, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 5: Broadcasting scalar beta to 3D x
    x = np.random.randn(2, 3, 4).astype(np.float32)
    beta = np.array(1.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 6: Integer types (x signed ints, beta positive ints)
    x = np.array([-3, -1, 0, 2, 4], dtype=np.int32)
    beta = np.array([1, 2, 3, 2, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 7: Small positive beta values (heavy-tailed)
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    beta = np.array([0.1, 0.25, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 8: Large beta values (approaching uniform distribution)
    x = np.linspace(-0.99, 0.99, 5, dtype=np.float64)
    beta = np.array([10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 9: 4D tensors with broadcasting
    x = np.random.randn(2, 1, 3, 1).astype(np.float32)
    beta = np.full((1, 4, 1, 5), 2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 10: Empty arrays
    x = np.array([], dtype=np.float64)
    beta = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 11: Zeros for x, mixed positive floats for beta
    x = np.zeros((4, 2), dtype=np.float32)
    beta = np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.logpdf_1"] = gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_1'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_1'], lib="jax", suffix=1)
