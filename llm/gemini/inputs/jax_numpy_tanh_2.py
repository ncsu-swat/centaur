
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import math

def tanh_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: positive one
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative one
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small positive value
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small negative value
    input_dict = {"x": -0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large positive value (saturation to 1.0)
    input_dict = {"x": 20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large negative value (saturation to -1.0)
    input_dict = {"x": -20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: very small positive float close to zero
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: pi
    input_dict = {"x": math.pi}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tanh_2"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tanh_2'.")


check_valid('jax.numpy.tanh', generated_inputs['jax.numpy.tanh_2'], lib="jax", suffix=2)
