
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import math

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard negative float
    input_dict = {"x": -42.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NaN
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large float
    input_dict = {"x": 1e308}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small positive float (subnormal range)
    input_dict = {"x": 1e-308}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mathematical constant pi
    input_dict = {"x": math.pi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative small float
    input_dict = {"x": -1.23456789e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_2"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_2'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_2'], lib="jax", suffix=2)
