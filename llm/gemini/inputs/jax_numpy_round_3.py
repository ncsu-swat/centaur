
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def round_inputs():
    list_of_inputs = []

    # Input 1: positive python int, decimals=0
    input_dict = {"a": 15, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative python int, decimals=1
    input_dict = {"a": -45, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: np.int32 positive, decimals=2
    input_dict = {"a": np.int32(123), "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int64 negative, decimals=0
    input_dict = {"a": np.int64(-987), "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: zero, decimals=5
    input_dict = {"a": 0, "decimals": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int32 positive, decimals=1
    input_dict = {"a": np.int32(50), "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int64 negative, decimals=3
    input_dict = {"a": np.int64(-7), "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: python int, decimals=0
    input_dict = {"a": 1000, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large python int, decimals=4
    input_dict = {"a": 1234567, "decimals": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative np.int32, decimals=2
    input_dict = {"a": np.int32(-88), "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.round_3"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.round_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.round_3'.")


check_valid('jax.numpy.round', generated_inputs['jax.numpy.round_3'], lib="jax", suffix=3)
