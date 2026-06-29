
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 2: Scalar values as 0D arrays
    q = np.array(0.4, dtype=np.float32)
    p = np.array(0.6, dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 3: 2D arrays with matching shapes
    q = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    p = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 4: Broadcasting shapes (1, 3) and (3, 1)
    q = np.array([[0.2, 0.5, 0.8]], dtype=np.float32)
    p = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 5: float64 precision 1D arrays
    q = np.array([0.05, 0.95], dtype=np.float64)
    p = np.array([0.2, 0.8], dtype=np.float64)
    list_of_inputs.append({"q": q, "p": p})

    # Input 6: Boundary values for q (0 and 1)
    q = np.array([0.0, 1.0], dtype=np.float32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 7: Boundary values for p (0 and 1)
    q = np.array([0.3, 0.7], dtype=np.float32)
    p = np.array([0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 8: 3D arrays
    q = np.random.uniform(0, 1, (2, 2, 2)).astype(np.float32)
    p = np.random.uniform(0, 1, (2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 9: Large 1D arrays
    q = np.linspace(0.01, 0.99, 100).astype(np.float32)
    p = np.full((100,), 0.3, dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    # Input 10: Broadcasting scalar 'p' with 2D 'q'
    q = np.random.uniform(0, 1, (4, 4)).astype(np.float32)
    p = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"q": q, "p": p})

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.ppf_1"] = jax_scipy_stats_bernoulli_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.ppf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.ppf_1'.")


check_valid('jax.scipy.stats.bernoulli.ppf', generated_inputs['jax.scipy.stats.bernoulli.ppf_1'], lib="jax", suffix=1)
