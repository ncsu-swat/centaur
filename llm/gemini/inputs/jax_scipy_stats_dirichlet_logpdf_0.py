
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dirichlet_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D size 3, float32
    x = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 2: 1D size 2 (K-1 for K=3), float32
    x = np.array([0.2, 0.5], dtype=np.float32)
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 3: 1D size 5, float32
    x = np.array([0.1, 0.2, 0.3, 0.1, 0.3], dtype=np.float32)
    alpha = np.array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 4: 1D size 4 (K-1 for K=5), float32
    x = np.array([0.1, 0.2, 0.3, 0.1], dtype=np.float32)
    alpha = np.array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 5: 1D size 2, float64
    x = np.array([0.5, 0.5], dtype=np.float64)
    alpha = np.array([0.5, 0.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 6: 1D size 1 (K-1 for K=2), float64
    x = np.array([0.4], dtype=np.float64)
    alpha = np.array([0.5, 0.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 7: 1D size 10, float32
    x = (np.ones(10) / 10.0).astype(np.float32)
    alpha = np.arange(1.0, 11.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 8: 1D size 9 (K-1 for K=10), float32
    x = (np.ones(9) / 10.0).astype(np.float32)
    alpha = np.arange(1.0, 11.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 9: Asymmetric alphas, size 4, float32
    x = np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32)
    alpha = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 10: Asymmetric alphas, size 3 (K-1 for K=4), float32
    x = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    alpha = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "alpha": alpha})

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
