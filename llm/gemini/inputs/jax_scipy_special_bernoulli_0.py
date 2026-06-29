
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bernoulli_inputs():
    list_of_inputs = []

    # Input 1: Very small positive integer
    input_dict = {"n": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {"n": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard size
    input_dict = {"n": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Medium size
    input_dict = {"n": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using numpy int32 type
    input_dict = {"n": np.int32(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger value
    input_dict = {"n": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using numpy int64 type
    input_dict = {"n": np.int64(30)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger value
    input_dict = {"n": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Even larger value
    input_dict = {"n": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another standard value
    input_dict = {"n": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Another standard value
    input_dict = {"n": 12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.bernoulli"] = bernoulli_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.bernoulli' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.bernoulli'.")


check_valid('jax.scipy.special.bernoulli', generated_inputs['jax.scipy.special.bernoulli'], lib="jax", suffix=0)
