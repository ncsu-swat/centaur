
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def less_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    list_of_inputs.append(copy.deepcopy({"x": 1.5, "y": 2.5}))

    # Input 2: Negative vs positive float
    list_of_inputs.append(copy.deepcopy({"x": -3.2, "y": 5.8}))

    # Input 3: Two negative floats
    list_of_inputs.append(copy.deepcopy({"x": -10.0, "y": -5.0}))

    # Input 4: Equal floats
    list_of_inputs.append(copy.deepcopy({"x": 42.0, "y": 42.0}))

    # Input 5: Zero and negative zero
    list_of_inputs.append(copy.deepcopy({"x": 0.0, "y": -0.0}))

    # Input 6: Small magnitude floats
    list_of_inputs.append(copy.deepcopy({"x": 1e-9, "y": 1e-8}))

    # Input 7: Large magnitude floats
    list_of_inputs.append(copy.deepcopy({"x": 1e12, "y": 1e11}))

    # Input 8: Positive infinity and finite float
    list_of_inputs.append(copy.deepcopy({"x": float("inf"), "y": 1000.0}))

    # Input 9: Negative infinity and finite float
    list_of_inputs.append(copy.deepcopy({"x": float("-inf"), "y": -1000.0}))

    # Input 10: Float with NaN value
    list_of_inputs.append(copy.deepcopy({"x": float("nan"), "y": 0.0}))

    # Input 11: Decimals close to each other
    list_of_inputs.append(copy.deepcopy({"x": 3.141592653589793, "y": 3.141592653589794}))

    return list_of_inputs

generated_inputs["jax.numpy.less_6"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_6'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_6'], lib="jax", suffix=6)
