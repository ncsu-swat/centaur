
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def poly_inputs():
    list_of_inputs = []

    # Input 1: Zero as integer
    input_dict = {"seq_of_zeros": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer 1
    input_dict = {"seq_of_zeros": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer -1
    input_dict = {"seq_of_zeros": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer
    input_dict = {"seq_of_zeros": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer
    input_dict = {"seq_of_zeros": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer
    input_dict = {"seq_of_zeros": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative integer
    input_dict = {"seq_of_zeros": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.int32 scalar
    input_dict = {"seq_of_zeros": np.int32(7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.int64 scalar
    input_dict = {"seq_of_zeros": np.int64(-15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int16 scalar
    input_dict = {"seq_of_zeros": np.int16(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: np.int8 scalar
    input_dict = {"seq_of_zeros": np.int8(-4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.poly_2"] = poly_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.poly_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.poly_2'.")


check_valid('jax.numpy.poly', generated_inputs['jax.numpy.poly_2'], lib="jax", suffix=2)
