
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dirichlet_pdf_inputs():
    list_of_inputs = []

    # Input 1: K=3, full simplex x (float32)
    raw_x = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    x = raw_x / np.sum(raw_x)
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 2: K=2, full simplex x (float64)
    raw_x = np.array([0.4, 0.6], dtype=np.float64)
    x = raw_x / np.sum(raw_x)
    alpha = np.array([0.5, 0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 3: K=3, reduced representation x of length K-1 (float32)
    x = np.array([0.2, 0.3], dtype=np.float32)
    alpha = np.array([2.0, 1.5, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 4: K=4, full simplex x (float64)
    raw_x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    x = raw_x / np.sum(raw_x)
    alpha = np.array([1.5, 2.5, 0.8, 3.2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 5: K=4, reduced representation x of length K-1 (float64)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    alpha = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 6: K=5, full simplex x with concentration params < 1 (float32)
    raw_x = np.array([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float32)
    x = raw_x / np.sum(raw_x)
    alpha = np.array([0.2, 0.4, 0.6, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 7: K=2, reduced representation x of length 1 (float32)
    x = np.array([0.35], dtype=np.float32)
    alpha = np.array([2.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 8: K=6, full simplex x with uniform alpha=1.0 (float64)
    raw_x = np.random.uniform(0.1, 1.0, size=(6,)).astype(np.float64)
    x = raw_x / np.sum(raw_x)
    alpha = np.ones(6, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 9: K=5, reduced representation x of length K-1 (float32)
    x = np.array([0.1, 0.15, 0.2, 0.25], dtype=np.float32)
    alpha = np.array([1.2, 2.3, 0.9, 1.8, 2.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 10: K=7, full simplex x with large concentration parameters (float64)
    raw_x = np.random.uniform(0.1, 1.0, size=(7,)).astype(np.float64)
    x = raw_x / np.sum(raw_x)
    alpha = np.array([10.0, 20.0, 15.0, 5.0, 12.0, 8.0, 25.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.dirichlet.pdf"] = dirichlet_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.dirichlet.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.dirichlet.pdf'.")


check_valid('jax.scipy.stats.dirichlet.pdf', generated_inputs['jax.scipy.stats.dirichlet.pdf'], lib="jax", suffix=0)
