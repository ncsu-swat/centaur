
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Float scalar representing integer 1
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float scalar representing integer 2
    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float scalar representing integer 3
    input_dict = {"x": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float scalar representing integer 5
    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float scalar representing integer 10
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy float32 scalar representing integer 4
    input_dict = {"x": np.float32(4.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy float32 scalar representing integer 8
    input_dict = {"x": np.float32(8.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy float64 scalar representing integer 20
    input_dict = {"x": np.float64(20.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy float64 scalar representing integer 50
    input_dict = {"x": np.float64(50.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy float32 scalar representing integer 100
    input_dict = {"x": np.float32(100.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.digamma_3"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_3'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_3'], lib="jax", suffix=3)
