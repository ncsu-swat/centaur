
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    k = np.array(1, dtype=np.int32)
    p = np.array(0.7, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 2: 1D arrays with k = 0
    k = np.array([0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.2, 0.5, 0.8, 0.1], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 3: Non-zero loc, float64
    k = np.array([2.0, 3.0], dtype=np.float64)
    p = np.array([0.4, 0.6], dtype=np.float64)
    loc = np.array([2.0, 2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 4: Negative loc
    k = np.array([-1, 0, -1], dtype=np.int32)
    p = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    loc = np.array([-1, -1, -1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 5: 2D arrays
    k = np.array([[0, 1], [1, 0]], dtype=np.int32)
    p = np.array([[0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 6: Broadcasting p across k
    k = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    p = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 7: Values of k outside the support {loc, loc + 1}
    k = np.array([0, 1, 2, -1], dtype=np.float32)
    p = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 8: 3D arrays
    k = np.ones((2, 3, 4), dtype=np.int32)
    p = np.full((2, 3, 4), 0.75, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 9: Boundary probabilities (p = 0.0, p = 1.0)
    k = np.array([0, 1], dtype=np.int32)
    p = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 10: Broadcasting loc across 2D array
    k = np.array([[1], [0], [1]], dtype=np.int32)
    p = np.array([[0.3, 0.6]], dtype=np.float32)
    loc = np.array([[0]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    # Input 11: Higher-precision float64 multidimensional
    k = np.array([[[1.0, 2.0]], [[2.0, 1.0]]], dtype=np.float64)
    p = np.array([[[0.25, 0.75]], [[0.5, 0.5]]], dtype=np.float64)
    loc = np.array([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "p": p, "loc": loc})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.bernoulli.logpmf"] = bernoulli_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.logpmf'.")


check_valid('jax.scipy.stats.bernoulli.logpmf', generated_inputs['jax.scipy.stats.bernoulli.logpmf'], lib="jax", suffix=0)
