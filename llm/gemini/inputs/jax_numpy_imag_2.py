
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def imag_inputs():
    list_of_inputs = []

    # Input 1: positive python integer
    input_dict = {"val": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative python integer
    input_dict = {"val": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero python integer
    input_dict = {"val": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large positive python integer
    input_dict = {"val": 100000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: large negative python integer
    input_dict = {"val": -50000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int32 positive
    input_dict = {"val": np.int32(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int32 negative
    input_dict = {"val": np.int32(-123)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.int32 zero
    input_dict = {"val": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.int64 positive
    input_dict = {"val": np.int64(999999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int64 negative
    input_dict = {"val": np.int64(-999999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: np.int64 zero
    input_dict = {"val": np.int64(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.imag_2"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.imag_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.imag_2'.")


check_valid('jax.numpy.imag', generated_inputs['jax.numpy.imag_2'], lib="jax", suffix=2)
