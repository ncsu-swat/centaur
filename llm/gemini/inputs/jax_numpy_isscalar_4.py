
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Python boolean True
    input_dict = {"element": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python boolean False
    input_dict = {"element": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NumPy boolean True
    input_dict = {"element": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NumPy boolean False
    input_dict = {"element": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero-dimensional array with True
    input_dict = {"element": np.array(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero-dimensional array with False
    input_dict = {"element": np.array(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero-dimensional array with explicit bool dtype
    input_dict = {"element": np.array(True, dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero-dimensional array with explicit np.bool_ dtype
    input_dict = {"element": np.array(False, dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NumPy boolean initialized with integer 1 (True)
    input_dict = {"element": np.bool_(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NumPy boolean initialized with integer 0 (False)
    input_dict = {"element": np.bool_(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_4"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_4'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_4'], lib="jax", suffix=4)
