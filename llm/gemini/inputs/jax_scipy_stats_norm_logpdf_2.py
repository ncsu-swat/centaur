
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def norm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard normal distribution at mean
    list_of_inputs.append({"x": 0.0, "loc": 0.0, "scale": 1.0})

    # Input 2: Positive value, standard parameters
    list_of_inputs.append({"x": 1.5, "loc": 0.0, "scale": 1.0})

    # Input 3: Negative value, standard parameters
    list_of_inputs.append({"x": -2.3, "loc": 0.0, "scale": 1.0})

    # Input 4: Value at positive mean with small scale
    list_of_inputs.append({"x": 5.0, "loc": 5.0, "scale": 0.5})

    # Input 5: Negative mean and negative value with scale > 1
    list_of_inputs.append({"x": -10.0, "loc": -5.0, "scale": 2.0})

    # Input 6: Large scale parameter
    list_of_inputs.append({"x": 0.0, "loc": -1.0, "scale": 10.0})

    # Input 7: Fractional values with small scale
    list_of_inputs.append({"x": 3.14159, "loc": 2.71828, "scale": 0.1})

    # Input 8: Very large scale parameter
    list_of_inputs.append({"x": -0.5, "loc": 0.5, "scale": 100.0})

    # Input 9: Large value with proportional scale
    list_of_inputs.append({"x": 100.0, "loc": 0.0, "scale": 50.0})

    # Input 10: Small magnitude values
    list_of_inputs.append({"x": 1e-3, "loc": 1e-4, "scale": 1e-2})

    # Input 11: Large negative offset matching x
    list_of_inputs.append({"x": -1000.0, "loc": -1000.0, "scale": 25.5})

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.logpdf_2"] = norm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.logpdf_2'.")


check_valid('jax.scipy.stats.norm.logpdf', generated_inputs['jax.scipy.stats.norm.logpdf_2'], lib="jax", suffix=2)
