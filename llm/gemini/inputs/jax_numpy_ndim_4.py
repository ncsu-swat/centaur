
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ndim_inputs():
    list_of_inputs = []

    # Input 1: Standard Python True
    input_dict = {"a": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard Python False
    input_dict = {"a": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NumPy scalar True
    input_dict = {"a": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy scalar False
    input_dict = {"a": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.True_ singleton
    input_dict = {"a": np.True_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.False_ singleton
    input_dict = {"a": np.False_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Casted boolean from integer 1
    input_dict = {"a": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Casted boolean from integer 0
    input_dict = {"a": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NumPy bool casted from integer 1
    input_dict = {"a": np.bool_(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NumPy bool casted from integer 0
    input_dict = {"a": np.bool_(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ndim_4"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_4'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_4'], lib="jax", suffix=4)
