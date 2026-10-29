
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shape_inputs():
    list_of_inputs = []

    # Input 1: standard python True
    input_dict = {"a": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard python False
    input_dict = {"a": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy bool True
    input_dict = {"a": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy bool False
    input_dict = {"a": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy True_ constant
    input_dict = {"a": np.True_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy False_ constant
    input_dict = {"a": np.False_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: boolean from comparison (True)
    input_dict = {"a": bool(1 < 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: boolean from comparison (False)
    input_dict = {"a": bool(2 < 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.bool_ from integer 1
    input_dict = {"a": np.bool_(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.bool_ from integer 0
    input_dict = {"a": np.bool_(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: extracted scalar from 0-d boolean array
    input_dict = {"a": np.array(True, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: extracted scalar from 0-d boolean array
    input_dict = {"a": np.array(False, dtype=bool).item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.shape_4"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_4'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_4'], lib="jax", suffix=4)
