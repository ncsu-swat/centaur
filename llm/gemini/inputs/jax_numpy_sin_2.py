
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_sin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Positive float
    list_of_inputs.append({"x": 1.0})

    # Input 3: Negative float
    list_of_inputs.append({"x": -1.0})

    # Input 4: Pi
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 5: Pi/2
    list_of_inputs.append({"x": 1.5707963267948966})

    # Input 6: -Pi/2
    list_of_inputs.append({"x": -1.5707963267948966})

    # Input 7: Small positive float
    list_of_inputs.append({"x": 1e-5})

    # Input 8: Small negative float
    list_of_inputs.append({"x": -1e-5})

    # Input 9: Large positive float
    list_of_inputs.append({"x": 100.0})

    # Input 10: Large negative float
    list_of_inputs.append({"x": -100.0})

    # Input 11: Arbitrary float
    list_of_inputs.append({"x": 12.34})

    return list_of_inputs


generated_inputs["jax.numpy.sin_2"] = jax_numpy_sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_2'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_2'], lib="jax", suffix=2)
