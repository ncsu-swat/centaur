
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.scipy.stats.gumbel_r

def gumbel_r_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Standard values
    list_of_inputs.append({"x": 0, "loc": 0, "scale": 1})

    # Input 2: Positive x, positive loc
    list_of_inputs.append({"x": 5, "loc": 2, "scale": 1})

    # Input 3: Negative x, zero loc, scale > 1
    list_of_inputs.append({"x": -3, "loc": 0, "scale": 2})

    # Input 4: Positive x, negative loc
    list_of_inputs.append({"x": 10, "loc": -5, "scale": 3})

    # Input 5: Negative x and negative loc
    list_of_inputs.append({"x": -10, "loc": -10, "scale": 5})

    # Input 6: Large positive values
    list_of_inputs.append({"x": 100, "loc": 50, "scale": 10})

    # Input 7: Negative x, positive loc
    list_of_inputs.append({"x": -50, "loc": 10, "scale": 4})

    # Input 8: Equal x and loc
    list_of_inputs.append({"x": 1, "loc": 1, "scale": 1})

    # Input 9: Moderate positive x, larger scale
    list_of_inputs.append({"x": 42, "loc": 0, "scale": 7})

    # Input 10: Extreme difference between x and loc
    list_of_inputs.append({"x": -100, "loc": 100, "scale": 20})

    # Input 11: Small scale, negative loc
    list_of_inputs.append({"x": 0, "loc": -1, "scale": 2})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logcdf_3"] = gumbel_r_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logcdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logcdf_3'.")


check_valid('jax.scipy.stats.gumbel_r.logcdf', generated_inputs['jax.scipy.stats.gumbel_r.logcdf_3'], lib="jax", suffix=3)
