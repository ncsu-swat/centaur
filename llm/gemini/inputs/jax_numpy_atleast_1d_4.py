
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_1d_inputs():
    list_of_inputs = []

    integers = [0, 1, -1, 42, -100, 255, 1024, -9999, 7, -42, 100000]

    for val in integers:
        input_dict = {"arys": int(val)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.atleast_1d_4"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_4'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_4'], lib="jax", suffix=4)
