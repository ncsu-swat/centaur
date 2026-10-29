
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic distribution at median
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive value, standard parameters
    input_dict = {"x": 2.5, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative value, standard parameters
    input_dict = {"x": -3.14159, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted distribution with loc > 0
    input_dict = {"x": 10.0, "loc": 10.0, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shifted distribution with loc < 0
    input_dict = {"x": -5.0, "loc": -5.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scale parameter
    input_dict = {"x": 50.0, "loc": 10.0, "scale": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scale parameter
    input_dict = {"x": 1.05, "loc": 1.0, "scale": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tail probability evaluation (large positive x)
    input_dict = {"x": 25.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tail probability evaluation (large negative x)
    input_dict = {"x": -30.0, "loc": 5.0, "scale": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Fractional values for all parameters
    input_dict = {"x": 0.123456, "loc": -0.789, "scale": 2.345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Non-zero location with small evaluation offset
    input_dict = {"x": 3.75, "loc": 3.5, "scale": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.pdf_2"] = logistic_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.pdf_2'.")


check_valid('jax.scipy.stats.logistic.pdf', generated_inputs['jax.scipy.stats.logistic.pdf_2'], lib="jax", suffix=2)
