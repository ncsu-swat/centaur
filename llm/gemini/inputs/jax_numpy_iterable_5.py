
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def jax_numpy_iterable_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"y": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"y": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"y": -2.71828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive float
    input_dict = {"y": 1e10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small positive float
    input_dict = {"y": 1e-12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative float
    input_dict = {"y": -3.5e8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative float
    input_dict = {"y": -5e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive infinity
    input_dict = {"y": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative infinity
    input_dict = {"y": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NaN (Not a Number)
    input_dict = {"y": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative zero
    input_dict = {"y": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: High precision float
    input_dict = {"y": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.iterable_5"] = jax_numpy_iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_5'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_5'], lib="jax", suffix=5)
