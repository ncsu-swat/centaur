
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_equal_6_inputs():
    list_of_inputs = []

    # Input 1: Basic equality with positive zero
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "y": 0.0}))

    # Input 2: Positive float equality
    list_of_inputs.append(copy.deepcopy({"x": 1.5, "y": 1.5}))

    # Input 3: Negative float equality
    list_of_inputs.append(copy.deepcopy({"x": -2.5, "y": -2.5}))

    # Input 4: Positive and negative zero comparison
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "y": -0.0}))

    # Input 5: Inequality with positive floats
    list_of_inputs.append(copy.deepcopy({"x": 1.23e-4, "y": 1.24e-4}))

    # Input 6: Large absolute value comparison (different signs)
    list_of_inputs.append(copy.deepcopy({"x": 9.87e6, "y": -9.87e6}))

    # Input 7: Infinity equality comparison
    list_of_inputs.append(copy.deepcopy({"x": float('inf'), "y": float('inf')}))

    # Input 8: Positive and negative infinity comparison
    list_of_inputs.append(copy.deepcopy({"x": float('-inf'), "y": float('inf')}))

    # Input 9: NaN comparison (NaN != NaN in standard IEEE 754)
    list_of_inputs.append(copy.deepcopy({"x": float('nan'), "y": float('nan')}))

    # Input 10: Comparison involving NaN and a regular float
    list_of_inputs.append(copy.deepcopy({"x": float('nan'), "y": 0.0}))

    # Input 11: Very small floating point values
    list_of_inputs.append(copy.deepcopy({"x": 1e-30, "y": 1e-30}))

    return list_of_inputs

generated_inputs["jax.numpy.equal_6"] = jax_numpy_equal_6_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_6'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_6'], lib="jax", suffix=6)
