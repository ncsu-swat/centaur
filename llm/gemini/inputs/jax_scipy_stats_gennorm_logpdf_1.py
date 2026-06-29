
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard normal-like (beta=2)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    beta = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, double exponential-like (beta=1)
    x = np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float32)
    beta = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 precision
    x = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    beta = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D random arrays
    x = np.random.randn(2, 2, 2).astype(np.float32)
    beta = (np.random.rand(2, 2, 2) + 0.5).astype(np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar beta (0D array) broadcasting with 1D x
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    beta = np.array(1.2, dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar x (0D array) broadcasting with 1D beta
    x = np.array(-0.5, dtype=np.float32)
    beta = np.array([0.8, 1.8, 2.8], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with different dimensions (3, 1) and (1, 4)
    x = np.random.randn(3, 1).astype(np.float32)
    beta = (np.random.rand(1, 4) * 3 + 0.1).astype(np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large beta values (approaching uniform distribution)
    x = np.array([-0.9, 0.0, 0.9], dtype=np.float32)
    beta = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small beta values
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    beta = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed integer type for x, float type for beta
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    beta = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
