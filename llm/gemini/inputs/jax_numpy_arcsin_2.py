
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arcsin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Positive value
    list_of_inputs.append({"x": 0.5})

    # Input 3: Negative value
    list_of_inputs.append({"x": -0.5})

    # Input 4: Upper boundary of domain
    list_of_inputs.append({"x": 1.0})

    # Input 5: Lower boundary of domain
    list_of_inputs.append({"x": -1.0})

    # Input 6: Small positive value
    list_of_inputs.append({"x": 0.12345})

    # Input 7: Small negative value
    list_of_inputs.append({"x": -0.12345})

    # Input 8: Value close to 1
    list_of_inputs.append({"x": 0.999})

    # Input 9: Value close to -1
    list_of_inputs.append({"x": -0.999})

    # Input 10: Value outside [-1, 1] (returns nan for real input, but still a valid float)
    list_of_inputs.append({"x": 1.5})

    # Input 11: Negative value outside [-1, 1]
    list_of_inputs.append({"x": -2.0})

    return list_of_inputs

generated_inputs["jax.numpy.arcsin_2"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsin_2'.")


check_valid('jax.numpy.arcsin', generated_inputs['jax.numpy.arcsin_2'], lib="jax", suffix=2)
