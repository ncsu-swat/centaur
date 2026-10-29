
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_ppf_inputs():
    list_of_inputs = []

    # Input 1: Standard median
    input_dict = {"q": float(0.5), "loc": float(0.0), "scale": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower tail 5%
    input_dict = {"q": float(0.05), "loc": float(0.0), "scale": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Upper tail 95%
    input_dict = {"q": float(0.95), "loc": float(0.0), "scale": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive loc, scale > 1
    input_dict = {"q": float(0.025), "loc": float(10.0), "scale": float(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc, scale < 1
    input_dict = {"q": float(0.975), "loc": float(-5.0), "scale": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative loc with large scale
    input_dict = {
        "q": float(0.5),
        "loc": float(-100.0),
        "scale": float(50.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small q with small scale
    input_dict = {"q": float(0.001), "loc": float(0.0), "scale": float(1e-3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High q with large loc and scale
    input_dict = {"q": float(0.999), "loc": float(1e3), "scale": float(1e2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Arbitrary float values
    input_dict = {
        "q": float(0.1),
        "loc": float(-0.75),
        "scale": float(3.14159),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Approximately 1-sigma upper quantile
    input_dict = {
        "q": float(0.8413447),
        "loc": float(0.0),
        "scale": float(1.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Approximately 1-sigma lower quantile with non-standard loc and scale
    input_dict = {"q": float(0.1586553), "loc": float(2.0), "scale": float(4.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Positive loc, tiny scale
    input_dict = {"q": float(0.3), "loc": float(1.5), "scale": float(0.01)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.ppf_2"] = norm_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.ppf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.ppf_2'.")


check_valid('jax.scipy.stats.norm.ppf', generated_inputs['jax.scipy.stats.norm.ppf_2'], lib="jax", suffix=2)
