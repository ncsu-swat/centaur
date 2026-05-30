
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: Positive integer as float
    list_of_inputs.append({"x": 2.0})

    # Input 2: Negative integer as float
    list_of_inputs.append({"x": -4.0})

    # Input 3: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: Fractional positive float
    list_of_inputs.append({"x": 0.5})

    # Input 5: Fractional negative float
    list_of_inputs.append({"x": -1.5})

    # Input 6: Larger positive float
    list_of_inputs.append({"x": 10.25})

    # Input 7: Larger negative float
    list_of_inputs.append({"x": -10.75})

    # Input 8: Float approximation of pi
    list_of_inputs.append({"x": float(np.pi)})

    # Input 9: Very small positive float
    list_of_inputs.append({"x": 1e-5})

    # Input 10: Negative zero
    list_of_inputs.append({"x": -0.0})

    return list_of_inputs

generated_inputs["jax.numpy.exp2_3"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp2_3'.")


check_valid('jax.numpy.exp2', generated_inputs['jax.numpy.exp2_3'], lib="jax", suffix=3)
