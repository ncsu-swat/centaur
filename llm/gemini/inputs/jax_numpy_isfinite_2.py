
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_isfinite_inputs():
    list_of_inputs = []

    # Input 1: positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: nan
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large float
    input_dict = {"x": 1.7976931348623157e308}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small float
    input_dict = {"x": 2.2250738585072014e-308}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy float64 scalar
    input_dict = {"x": np.float64(3.14159)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: numpy float32 scalar
    input_dict = {"x": np.float32(-0.001)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_2"] = jax_numpy_isfinite_inputs()

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
