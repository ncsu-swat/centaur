
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard positive integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Standard negative integer
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 32-bit positive integer boundary
    input_dict = {"x": 2147483647}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 32-bit negative integer boundary
    input_dict = {"x": -2147483648}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Medium positive integer
    input_dict = {"x": 65535}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive 64-bit integer
    input_dict = {"x": 1000000000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative 64-bit integer
    input_dict = {"x": -1000000000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Odd positive integer
    input_dict = {"x": 9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.iscomplexobj_3"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_3'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_3'], lib="jax", suffix=3)
