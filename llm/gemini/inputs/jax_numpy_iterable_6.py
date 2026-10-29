
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iterable_inputs():
    list_of_inputs = []

    # Input 1: Python True
    input_dict = {"y": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python False
    input_dict = {"y": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy bool True
    input_dict = {"y": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy bool False
    input_dict = {"y": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Python bool from int 1
    input_dict = {"y": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Python bool from int 0
    input_dict = {"y": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy True_ constant
    input_dict = {"y": np.True_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy False_ constant
    input_dict = {"y": np.False_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Python bool from comparison
    input_dict = {"y": bool(2 > 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Python bool from comparison
    input_dict = {"y": bool(1 > 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.iterable_6"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_6'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_6'], lib="jax", suffix=6)
