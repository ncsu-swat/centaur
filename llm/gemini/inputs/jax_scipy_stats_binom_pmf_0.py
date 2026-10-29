
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def binom_pmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    list_of_inputs.append(
        {
            "k": np.array(3, dtype=np.int32),
            "n": np.array(10, dtype=np.int32),
            "p": np.array(0.5, dtype=np.float32),
            "loc": np.array(0, dtype=np.int32),
        }
    )

    # Input 2: 1D arrays, same shape
    list_of_inputs.append(
        {
            "k": np.array([0, 1, 2, 3], dtype=np.float32),
            "n": np.array([5, 5, 5, 5], dtype=np.float32),
            "p": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    )

    # Input 3: Non-zero loc (positive loc)
    list_of_inputs.append(
        {
            "k": np.array([2, 3, 4, 5], dtype=np.int32),
            "n": np.array([5, 5, 5, 5], dtype=np.int32),
            "p": np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
            "loc": np.array([2, 2, 2, 2], dtype=np.int32),
        }
    )

    # Input 4: Negative loc
    list_of_inputs.append(
        {
            "k": np.array([-1, 0, 1, 2], dtype=np.int64),
            "n": np.array([4, 4, 4, 4], dtype=np.int64),
            "p": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
            "loc": np.array([-2, -2, -2, -2], dtype=np.int64),
        }
    )

    # Input 5: 2D arrays
    list_of_inputs.append(
        {
            "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "n": np.array([[5, 5], [5, 5]], dtype=np.int32),
            "p": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "loc": np.array([[0, 0], [0, 0]], dtype=np.int32),
        }
    )

    # Input 6: Broadcasting between k, n, p, loc
    list_of_inputs.append(
        {
            "k": np.arange(6, dtype=np.float32).reshape(6, 1),
            "n": np.array([10], dtype=np.float32),
            "p": np.array([0.2, 0.5, 0.8], dtype=np.float32).reshape(1, 3),
            "loc": np.array([0.0], dtype=np.float32),
        }
    )

    # Input 7: Edge probabilities p=0.0 and p=1.0
    list_of_inputs.append(
        {
            "k": np.array([0, 10], dtype=np.int32),
            "n": np.array([10, 10], dtype=np.int32),
            "p": np.array([0.0, 1.0], dtype=np.float32),
            "loc": np.array([0, 0], dtype=np.int32),
        }
    )

    # Input 8: Values of k outside valid support (k < loc or k > n + loc)
    list_of_inputs.append(
        {
            "k": np.array([-1, 0, 5, 11], dtype=np.float64),
            "n": np.array([10, 10, 10, 10], dtype=np.float64),
            "p": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
    )

    # Input 9: 3D tensor inputs
    list_of_inputs.append(
        {
            "k": np.ones((2, 3, 4), dtype=np.int32) * 2,
            "n": np.ones((2, 3, 4), dtype=np.int32) * 6,
            "p": np.full((2, 3, 4), 0.25, dtype=np.float32),
            "loc": np.zeros((2, 3, 4), dtype=np.int32),
        }
    )

    # Input 10: Float64 precision with broadcast loc
    list_of_inputs.append(
        {
            "k": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64),
            "n": np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64),
            "p": np.array([[0.5], [0.5]], dtype=np.float64),
            "loc": np.array([1.0], dtype=np.float64),
        }
    )

    # Input 11: n=0 trials
    list_of_inputs.append(
        {
            "k": np.array([0, 1], dtype=np.int32),
            "n": np.array([0, 0], dtype=np.int32),
            "p": np.array([0.5, 0.5], dtype=np.float32),
            "loc": np.array([0, 0], dtype=np.int32),
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.binom.pmf"] = binom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.binom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.binom.pmf'.")


check_valid('jax.scipy.stats.binom.pmf', generated_inputs['jax.scipy.stats.binom.pmf'], lib="jax", suffix=0)
