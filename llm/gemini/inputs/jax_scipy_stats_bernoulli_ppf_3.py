
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar integers
    list_of_inputs.append({"q": 0, "p": 1})

    # Input 2: Basic scalar integers reversed
    list_of_inputs.append({"q": 1, "p": 0})

    # Input 3: np.int32 scalars
    list_of_inputs.append({"q": np.int32(0), "p": np.int32(0)})

    # Input 4: np.int64 scalars
    list_of_inputs.append({"q": np.int64(1), "p": np.int64(1)})

    # Input 5: 1D numpy array of int32
    list_of_inputs.append({
        "q": np.array([0, 1], dtype=np.int32),
        "p": np.array([1, 0], dtype=np.int32)
    })

    # Input 6: 2D numpy array of int64
    list_of_inputs.append({
        "q": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "p": np.array([[1, 1], [0, 0]], dtype=np.int64)
    })

    # Input 7: np.int16 scalars
    list_of_inputs.append({"q": np.int16(1), "p": np.int16(0)})

    # Input 8: np.int8 scalars
    list_of_inputs.append({"q": np.int8(0), "p": np.int8(1)})

    # Input 9: Mixed integer scalar types
    list_of_inputs.append({"q": np.int32(1), "p": np.int64(0)})

    # Input 10: 1D numpy array of int16
    list_of_inputs.append({
        "q": np.array([1, 1, 0], dtype=np.int16),
        "p": np.array([0, 1, 1], dtype=np.int16)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.ppf_3"] = jax_scipy_stats_bernoulli_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.ppf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.ppf_3'.")


check_valid('jax.scipy.stats.bernoulli.ppf', generated_inputs['jax.scipy.stats.bernoulli.ppf_3'], lib="jax", suffix=3)
