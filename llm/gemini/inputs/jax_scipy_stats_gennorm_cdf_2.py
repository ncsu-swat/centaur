
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard normal equivalent (beta = 2.0) at mean
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "beta": 2.0}))

    # Input 2: Laplace distribution equivalent (beta = 1.0), positive x
    list_of_inputs.append(copy.deepcopy({"x": 1.0, "beta": 1.0}))

    # Input 3: Negative x with beta = 2.0
    list_of_inputs.append(copy.deepcopy({"x": -1.5, "beta": 2.0}))

    # Input 4: Heavy-tailed distribution (beta < 1)
    list_of_inputs.append(copy.deepcopy({"x": 2.5, "beta": 0.5}))

    # Input 5: Platykurtic / lighter-tailed distribution (beta > 2)
    list_of_inputs.append(copy.deepcopy({"x": -3.0, "beta": 5.0}))

    # Input 6: Non-integer beta between 1 and 2
    list_of_inputs.append(copy.deepcopy({"x": 0.75, "beta": 1.5}))

    # Input 7: Small positive beta, small negative x
    list_of_inputs.append(copy.deepcopy({"x": -0.1, "beta": 0.25}))

    # Input 8: Large positive x with beta = 3.0
    list_of_inputs.append(copy.deepcopy({"x": 10.0, "beta": 3.0}))

    # Input 9: Large negative x with standard normal shape
    list_of_inputs.append(copy.deepcopy({"x": -10.0, "beta": 2.0}))

    # Input 10: Approaching uniform distribution (large beta)
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "beta": 10.0}))

    # Input 11: Small positive x close to zero
    list_of_inputs.append(copy.deepcopy({"x": 1e-4, "beta": 2.5}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.cdf_2"] = gennorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.cdf_2'.")


check_valid('jax.scipy.stats.gennorm.cdf', generated_inputs['jax.scipy.stats.gennorm.cdf_2'], lib="jax", suffix=2)
