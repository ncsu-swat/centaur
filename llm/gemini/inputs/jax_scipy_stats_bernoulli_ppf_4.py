
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: standard Python booleans (True, True)
    input_dict = {"q": True, "p": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard Python booleans (False, True)
    input_dict = {"q": False, "p": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: standard Python booleans (True, False)
    input_dict = {"q": True, "p": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: standard Python booleans (False, False)
    input_dict = {"q": False, "p": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy boolean scalars (True, True)
    input_dict = {"q": np.bool_(True), "p": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy boolean scalars (False, True)
    input_dict = {"q": np.bool_(False), "p": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy boolean scalars (True, False)
    input_dict = {"q": np.bool_(True), "p": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy boolean scalars (False, False)
    input_dict = {"q": np.bool_(False), "p": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: mixed Python bool and numpy boolean (True, False)
    input_dict = {"q": True, "p": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: mixed numpy boolean and Python bool (False, True)
    input_dict = {"q": np.bool_(False), "p": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: mixed numpy boolean and Python bool (True, True)
    input_dict = {"q": np.bool_(True), "p": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: mixed Python bool and numpy boolean (False, False)
    input_dict = {"q": False, "p": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.bernoulli.ppf_4"] = bernoulli_ppf_inputs()

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
