
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vonmises_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard values around zero
    input_dict = {"x": 0.0, "kappa": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero concentration (uniform distribution on circle)
    input_dict = {"x": float(np.pi / 4), "kappa": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative angle, moderate concentration
    input_dict = {"x": -float(np.pi / 2), "kappa": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive boundary angle pi
    input_dict = {"x": float(np.pi), "kappa": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative boundary angle -pi
    input_dict = {"x": -float(np.pi), "kappa": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: High concentration parameter
    input_dict = {"x": 0.1, "kappa": 50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small concentration parameter
    input_dict = {"x": -1.25, "kappa": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Peak value with moderate kappa
    input_dict = {"x": 0.0, "kappa": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Arbitrary positive angle
    input_dict = {"x": 2.356194490192345, "kappa": 1.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arbitrary negative angle
    input_dict = {"x": -0.7853981633974483, "kappa": 4.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Angle beyond [-pi, pi] (periodic evaluation)
    input_dict = {"x": float(3 * np.pi), "kappa": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.vonmises.pdf_2"] = vonmises_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.vonmises.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.vonmises.pdf_2'.")


check_valid('jax.scipy.stats.vonmises.pdf', generated_inputs['jax.scipy.stats.vonmises.pdf_2'], lib="jax", suffix=2)
