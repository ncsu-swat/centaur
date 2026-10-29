
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax

def lax_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    list_of_inputs.append(copy.deepcopy({"x": 5.0, "y": 2.0}))

    # Input 2: Zero values
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "y": 0.0}))

    # Input 3: Negative and positive float
    list_of_inputs.append(copy.deepcopy({"x": -3.5, "y": 4.5}))

    # Input 4: Both negative floats
    list_of_inputs.append(copy.deepcopy({"x": -10.25, "y": -5.75}))

    # Input 5: Small fractional floats
    list_of_inputs.append(copy.deepcopy({"x": 1e-5, "y": 2e-5}))

    # Input 6: Large floats
    list_of_inputs.append(copy.deepcopy({"x": 1e8, "y": 5e7}))

    # Input 7: Subtracting larger from smaller
    list_of_inputs.append(copy.deepcopy({"x": 1.25, "y": 100.5}))

    # Input 8: High precision mathematical constants
    list_of_inputs.append(copy.deepcopy({"x": 3.141592653589793, "y": 2.718281828459045}))

    # Input 9: Negative zero and positive float
    list_of_inputs.append(copy.deepcopy({"x": -0.0, "y": 15.0}))

    # Input 10: Subtraction resulting in zero
    list_of_inputs.append(copy.deepcopy({"x": 42.42, "y": 42.42}))

    # Input 11: Very small difference
    list_of_inputs.append(copy.deepcopy({"x": 1.0000001, "y": 1.0}))

    return list_of_inputs

generated_inputs["jax.lax.sub_3"] = lax_sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_3'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_3'], lib="jax", suffix=3)
