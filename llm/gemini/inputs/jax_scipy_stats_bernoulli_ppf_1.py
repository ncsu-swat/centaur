
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    q = np.array(0.5, dtype=np.float32)
    p = np.array(0.3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 2: 1D arrays of matching shape
    q = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 3: Float64 precision
    q = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float64)
    p = np.array([0.2, 0.3, 0.6, 0.8], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 4: 2D arrays with identical shape
    q = np.array([[0.1, 0.6], [0.3, 0.9]], dtype=np.float32)
    p = np.array([[0.4, 0.4], [0.7, 0.7]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 5: Broadcasting 1D to 2D
    q = np.array([[0.2], [0.5], [0.8]], dtype=np.float32)
    p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 6: Boundary values of p (0.0 and 1.0)
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    p = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 7: Out-of-bounds quantiles (q < 0 and q > 1)
    q = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    p = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 8: 3D arrays
    q = np.random.uniform(0.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    p = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 9: Scalar q broadcasted over 2D p
    q = np.array(0.4, dtype=np.float64)
    p = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 10: Empty arrays
    q = np.array([], dtype=np.float32)
    p = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    # Input 11: Broadcasting (1, 4) and (3, 1)
    q = np.array([[0.1, 0.3, 0.7, 0.9]], dtype=np.float32)
    p = np.array([[0.2], [0.5], [0.8]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "p": p}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.bernoulli.ppf_1"] = bernoulli_ppf_inputs()

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
