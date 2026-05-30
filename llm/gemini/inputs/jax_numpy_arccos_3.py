
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arccos_inputs():
    list_of_inputs = []

    # Input 1: Zero (int)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive one (int)
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative one (int)
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer outside [-1, 1] range (int)
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer outside [-1, 1] range (int)
    input_dict = {"x": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive integer (int)
    input_dict = {"x": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative integer (int)
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy int32 zero
    input_dict = {"x": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy int64 one
    input_dict = {"x": np.int64(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy int16 negative one
    input_dict = {"x": np.int16(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: numpy int8 positive integer
    input_dict = {"x": np.int8(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: numpy int32 negative integer
    input_dict = {"x": np.int32(-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arccos_3"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccos_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccos_3'.")


check_valid('jax.numpy.arccos', generated_inputs['jax.numpy.arccos_3'], lib="jax", suffix=3)
