
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sqrt_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float
    list_of_inputs.append({"x": 4.0})

    # Input 2: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 3: Negative float
    list_of_inputs.append({"x": -1.0})

    # Input 4: Large float
    list_of_inputs.append({"x": 1e10})

    # Input 5: Small positive float
    list_of_inputs.append({"x": 1e-10})

    # Input 6: Fraction as float
    list_of_inputs.append({"x": 0.25})

    # Input 7: Float approximation of Pi
    list_of_inputs.append({"x": 3.1415926535})

    # Input 8: numpy.float32 scalar
    list_of_inputs.append({"x": np.float32(9.0).item()})

    # Input 9: numpy.float64 scalar
    list_of_inputs.append({"x": np.float64(16.0).item()})

    # Input 10: Infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 11: Large decimal float
    list_of_inputs.append({"x": 123456.789})

    return list_of_inputs

generated_inputs["jax.numpy.sqrt_3"] = jax_numpy_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_3'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_3'], lib="jax", suffix=3)
