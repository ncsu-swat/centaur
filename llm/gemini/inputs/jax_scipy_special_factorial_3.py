
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_factorial_inputs():
    list_of_inputs = []

    # Input 1: Small integer float
    list_of_inputs.append({"n": 0.0, "exact": False})

    # Input 2: Integer float 1.0
    list_of_inputs.append({"n": 1.0, "exact": False})

    # Input 3: Integer float 5.0
    list_of_inputs.append({"n": 5.0, "exact": False})

    # Input 4: Larger integer float
    list_of_inputs.append({"n": 10.0, "exact": False})

    # Input 5: Non-integer float 2.5
    list_of_inputs.append({"n": 2.5, "exact": False})

    # Input 6: Non-integer float 4.7
    list_of_inputs.append({"n": 4.7, "exact": False})

    # Input 7: Fractional float 0.5
    list_of_inputs.append({"n": 0.5, "exact": False})

    # Input 8: Larger non-integer float
    list_of_inputs.append({"n": 12.3, "exact": False})

    # Input 9: Floating point value for 20.0
    list_of_inputs.append({"n": 20.0, "exact": False})

    # Input 10: Mathematical constant approximation (pi)
    list_of_inputs.append({"n": 3.14159, "exact": False})

    # Input 11: Negative non-integer float (supported via gamma function)
    list_of_inputs.append({"n": -0.5, "exact": False})

    return list_of_inputs

generated_inputs["jax.scipy.special.factorial_3"] = jax_scipy_special_factorial_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.factorial_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.factorial_3'.")


check_valid('jax.scipy.special.factorial', generated_inputs['jax.scipy.special.factorial_3'], lib="jax", suffix=3)
