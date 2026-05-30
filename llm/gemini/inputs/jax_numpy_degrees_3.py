
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Python int zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python int positive
    input_dict = {"x": int(90)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Python int negative
    input_dict = {"x": int(-180)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int32 positive
    input_dict = {"x": np.int32(45)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.int32 negative
    input_dict = {"x": np.int32(-60)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.int32 zero
    input_dict = {"x": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.int64 positive
    input_dict = {"x": np.int64(360)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.int64 negative
    input_dict = {"x": np.int64(-270)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Python int large
    input_dict = {"x": int(1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int64 zero
    input_dict = {"x": np.int64(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.degrees_3"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_3'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_3'], lib="jax", suffix=3)
