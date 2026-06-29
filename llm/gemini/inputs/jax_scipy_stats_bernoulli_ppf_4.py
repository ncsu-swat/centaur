
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "q": np.bool_(True),
        "p": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "q": np.bool_(False),
        "p": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "q": np.array([True, False, True], dtype=np.bool_),
        "p": np.array([False, True, True], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "q": np.array([[True, False], [False, True]], dtype=np.bool_),
        "p": np.array([[False, True], [True, False]], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "q": np.bool_(True),
        "p": np.array([True, False, True], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "q": np.array([False, False, True], dtype=np.bool_),
        "p": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "q": np.ones((2, 2, 2), dtype=np.bool_),
        "p": np.zeros((2, 2, 2), dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "q": np.array([True], dtype=np.bool_),
        "p": np.array([True, False], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "q": np.ones((3, 3), dtype=np.bool_),
        "p": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "q": np.zeros((2, 2, 1, 2), dtype=np.bool_),
        "p": np.ones((2, 2, 1, 2), dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.ppf_4"] = jax_scipy_stats_bernoulli_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.ppf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.ppf_4'.")


check_valid('jax.scipy.stats.bernoulli.ppf', generated_inputs['jax.scipy.stats.bernoulli.ppf_4'], lib="jax", suffix=4)
