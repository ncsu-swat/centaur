
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dirichlet_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D tensors, K=3, float32
    x = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 2: 1D tensors, K=2, float64
    x = np.array([0.5, 0.5], dtype=np.float64)
    alpha = np.array([0.5, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 3: 1D tensors, K=4, exact binary fractions
    x = np.array([0.125, 0.25, 0.125, 0.5], dtype=np.float32)
    alpha = np.array([2.5, 1.5, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 4: 2D x with shape (K, N) where axis 0 is K=3, 1D alpha of shape (3,)
    x_raw = np.array(
        [[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 1.0, 2.0], [1.0, 1.0, 2.0, 4.0]],
        dtype=np.float32,
    )
    x = x_raw / np.sum(x_raw, axis=0, keepdims=True)
    alpha = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 5: 2D x with shape (K, N) where axis 0 is K=2, 1D alpha of shape (2,)
    x_raw = np.array([[0.3, 0.8, 0.5], [0.7, 0.2, 0.5]], dtype=np.float64)
    x = x_raw / np.sum(x_raw, axis=0, keepdims=True)
    alpha = np.array([2.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 6: 1D tensors, K=5, uniform Dirichlet (alpha = 1)
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    alpha = np.ones(5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 7: 3D x with shape (K, N, M) where axis 0 is K=4, 1D alpha of shape (4,)
    x_raw = np.ones((4, 2, 3), dtype=np.float32)
    x = x_raw / np.sum(x_raw, axis=0, keepdims=True)
    alpha = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 8: 1D x with K-1 entries (implicit remaining component), alpha of shape (3,)
    x = np.array([0.2, 0.3], dtype=np.float64)
    alpha = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 9: 1D tensors, alpha with sparse parameters (< 1.0)
    x = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    alpha = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 10: 1D tensors, large alpha values (concentrated distribution)
    x = np.array([0.4, 0.35, 0.25], dtype=np.float64)
    alpha = np.array([100.0, 87.5, 62.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.dirichlet.logpdf"] = dirichlet_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.dirichlet.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.dirichlet.logpdf'.")


check_valid('jax.scipy.stats.dirichlet.logpdf', generated_inputs['jax.scipy.stats.dirichlet.logpdf'], lib="jax", suffix=0)
