
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def le_inputs():
    list_of_inputs = []

    # Input 1: python bool (True, True)
    input_dict = {"x": True, "y": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: python bool (True, False)
    input_dict = {"x": True, "y": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: python bool (False, True)
    input_dict = {"x": False, "y": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: python bool (False, False)
    input_dict = {"x": False, "y": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy bool_ (True, True)
    input_dict = {"x": np.bool_(True), "y": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy bool_ (True, False)
    input_dict = {"x": np.bool_(True), "y": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy bool_ (False, True)
    input_dict = {"x": np.bool_(False), "y": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy bool_ (False, False)
    input_dict = {"x": np.bool_(False), "y": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: mixed bool and numpy bool_ (True, np.bool_(False))
    input_dict = {"x": True, "y": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: mixed bool and numpy bool_ (np.bool_(True), False)
    input_dict = {"x": np.bool_(True), "y": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: mixed bool and numpy bool_ (False, np.bool_(True))
    input_dict = {"x": False, "y": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: mixed bool and numpy bool_ (np.bool_(False), True)
    input_dict = {"x": np.bool_(False), "y": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.le_4"] = le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_4'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_4'], lib="jax", suffix=4)
