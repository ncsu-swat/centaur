
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_scipy_stats_cauchy_logsf_inputs():
    list_of_inputs = []

    # Input 1: Standard Cauchy, x=0
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard Cauchy, positive x
    input_dict = {"x": 1.5, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard Cauchy, negative x
    input_dict = {"x": -1.5, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted and scaled, positive x
    input_dict = {"x": 10.0, "loc": 2.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shifted and scaled, negative x
    input_dict = {"x": -5.0, "loc": -1.0, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scale
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scale, positive offset
    input_dict = {"x": 100.0, "loc": 50.0, "scale": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Standard scale, large negative x
    input_dict = {"x": -100.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small scale, x equal to loc
    input_dict = {"x": 0.5, "loc": 0.5, "scale": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: General values
    input_dict = {"x": -0.2, "loc": 1.2, "scale": 3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logsf_3"] = (
    jax_scipy_stats_cauchy_logsf_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logsf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logsf_3'.")


check_valid('jax.scipy.stats.cauchy.logsf', generated_inputs['jax.scipy.stats.cauchy.logsf_3'], lib="jax", suffix=3)
