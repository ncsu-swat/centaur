
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def expon_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard exponential evaluation at 1.0
    input_dict = {"x": 1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Boundary point x == loc
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x < loc (outside support)
    input_dict = {"x": -1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive loc, small scale
    input_dict = {"x": 2.5, "loc": 1.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc, large scale
    input_dict = {"x": 5.0, "loc": -2.0, "scale": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Fractional values
    input_dict = {"x": 0.5, "loc": 0.1, "scale": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values
    input_dict = {"x": 100.0, "loc": 50.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small scale parameter
    input_dict = {"x": 0.001, "loc": 0.0, "scale": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative loc with x >= loc
    input_dict = {"x": -5.0, "loc": -10.0, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-integer floats
    input_dict = {"x": 3.14159, "loc": 2.71828, "scale": 1.41421}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x strictly less than loc with positive loc
    input_dict = {"x": 10.0, "loc": 15.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.pdf_3"] = expon_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.pdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.pdf_3'.")


check_valid('jax.scipy.stats.expon.pdf', generated_inputs['jax.scipy.stats.expon.pdf_3'], lib="jax", suffix=3)
