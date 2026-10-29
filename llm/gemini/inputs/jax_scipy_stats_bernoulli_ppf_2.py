
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: Median with fair coin
    input_dict = {"q": 0.5, "p": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower quantile with fair coin
    input_dict = {"q": 0.1, "p": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Upper quantile with fair coin
    input_dict = {"q": 0.9, "p": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Minimum quantile (0.0)
    input_dict = {"q": 0.0, "p": 0.3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Maximum quantile (1.0)
    input_dict = {"q": 1.0, "p": 0.7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Highly biased towards 1
    input_dict = {"q": 0.05, "p": 0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Highly biased towards 0
    input_dict = {"q": 0.95, "p": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boundary parameter p = 0.0
    input_dict = {"q": 0.5, "p": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boundary parameter p = 1.0
    input_dict = {"q": 0.5, "p": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Transition point q = 1 - p
    input_dict = {"q": 0.6, "p": 0.4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Just below transition point
    input_dict = {"q": 0.39, "p": 0.6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Just above transition point
    input_dict = {"q": 0.41, "p": 0.6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.ppf_2"] = bernoulli_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.ppf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.ppf_2'.")


check_valid('jax.scipy.stats.bernoulli.ppf', generated_inputs['jax.scipy.stats.bernoulli.ppf_2'], lib="jax", suffix=2)
