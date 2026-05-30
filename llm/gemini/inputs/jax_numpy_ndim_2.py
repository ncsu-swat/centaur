
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndim_inputs():
    list_of_inputs = []

    list_of_inputs.append({"a": 1})
    list_of_inputs.append({"a": 0})
    list_of_inputs.append({"a": -1})
    list_of_inputs.append({"a": 100})
    list_of_inputs.append({"a": -500})
    list_of_inputs.append({"a": 123456})
    list_of_inputs.append({"a": -987654})
    list_of_inputs.append({"a": 10})
    list_of_inputs.append({"a": -20})
    list_of_inputs.append({"a": 2024})
    list_of_inputs.append({"a": -32768})
    list_of_inputs.append({"a": 65535})

    return list_of_inputs

generated_inputs["jax.numpy.ndim_2"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_2'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_2'], lib="jax", suffix=2)
