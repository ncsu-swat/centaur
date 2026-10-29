
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"a": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float
    input_dict = {"a": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"a": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard mathematical constant approximation
    input_dict = {"a": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive float
    input_dict = {"a": 1e10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive float
    input_dict = {"a": 1e-8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative float
    input_dict = {"a": -1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive infinity
    input_dict = {"a": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative infinity
    input_dict = {"a": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NaN (Not a Number)
    input_dict = {"a": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary precision decimal
    input_dict = {"a": 42.4242}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.shape_3"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_3'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_3'], lib="jax", suffix=3)
