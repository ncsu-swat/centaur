
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard beta=2.0 (normal distribution equivalent)
    x = np.array(0.0, dtype=np.float32)
    beta = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 2: 1D arrays of matching shape, Laplace-like distribution (beta=1.0)
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 3: 1D array for x and scalar array for beta (broadcasting)
    x = np.linspace(-5.0, 5.0, 20, dtype=np.float64)
    beta = np.array(0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 4: 2D arrays with identical shape
    x = np.array([[-1.5, 0.2], [0.8, -2.1]], dtype=np.float32)
    beta = np.array([[1.5, 2.5], [3.0, 0.8]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 5: 2D array x broadcast with 1D array beta
    x = np.ones((3, 4), dtype=np.float32) * -0.5
    beta = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 6: 3D arrays with float64 precision
    x = np.zeros((2, 3, 4), dtype=np.float64)
    beta = np.full((2, 3, 4), 3.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 7: Large beta values approaching uniform distribution
    x = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    beta = np.array([10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 8: Small beta values (sharp peak at 0)
    x = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    beta = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 9: Multi-dimensional broadcasting with singleton dimensions
    x = np.arange(-3, 3, dtype=np.float32).reshape(6, 1)
    beta = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    # Input 10: 4D arrays with extreme positive and negative values in x
    x = np.array([[[[-100.0]], [[0.0]], [[100.0]]]], dtype=np.float32)
    beta = np.array([[[[2.0]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "beta": beta}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.pdf"] = gennorm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.pdf'.")


check_valid('jax.scipy.stats.gennorm.pdf', generated_inputs['jax.scipy.stats.gennorm.pdf'], lib="jax", suffix=0)
