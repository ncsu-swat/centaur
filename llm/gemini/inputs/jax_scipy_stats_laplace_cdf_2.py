
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def laplace_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Laplace CDF at median
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Right tail value
    input_dict = {"x": 2.5, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Left tail value
    input_dict = {"x": -3.0, "loc": 0.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive location, larger scale
    input_dict = {"x": 10.0, "loc": 5.0, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative location and x
    input_dict = {"x": -5.0, "loc": -2.0, "scale": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small scale (peaked distribution)
    input_dict = {"x": 1.05, "loc": 1.0, "scale": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Far in the right tail
    input_dict = {"x": 100.0, "loc": 10.0, "scale": 15.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Far in the left tail
    input_dict = {"x": -100.0, "loc": 0.0, "scale": 20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x strictly equals loc
    input_dict = {"x": -4.2, "loc": -4.2, "scale": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Fractional values
    input_dict = {"x": 0.123, "loc": -0.456, "scale": 1.789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large scale (flat distribution)
    input_dict = {"x": 15.0, "loc": 0.0, "scale": 500.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.laplace.cdf_2"] = laplace_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.cdf_2'.")


check_valid('jax.scipy.stats.laplace.cdf', generated_inputs['jax.scipy.stats.laplace.cdf_2'], lib="jax", suffix=2)
