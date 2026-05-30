
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def poly_inputs():
    list_of_inputs = []

    # Input 1: positive float
    input_dict = {"seq_of_zeros": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float
    input_dict = {"seq_of_zeros": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero float
    input_dict = {"seq_of_zeros": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.float32 scalar
    input_dict = {"seq_of_zeros": np.float32(3.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.float64 scalar
    input_dict = {"seq_of_zeros": np.float64(-0.123)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large float
    input_dict = {"seq_of_zeros": 1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: small float
    input_dict = {"seq_of_zeros": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative large float
    input_dict = {"seq_of_zeros": -500.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: pi-like float
    input_dict = {"seq_of_zeros": 3.1415926535}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: e-like float
    input_dict = {"seq_of_zeros": 2.7182818284}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative small float
    input_dict = {"seq_of_zeros": -1.2e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.poly_3"] = poly_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.poly_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.poly_3'.")


check_valid('jax.numpy.poly', generated_inputs['jax.numpy.poly_3'], lib="jax", suffix=3)
