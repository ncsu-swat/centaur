
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def kaiser_inputs():
    list_of_inputs = []

    # Input 1, valid — float32 beta
    input_dict = {
        "M": 4,
        "beta": np.array(1.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — larger M
    input_dict = {
        "M": 10,
        "beta": np.array(5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — M is 0
    input_dict = {
        "M": 0,
        "beta": np.array(5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — float64 beta, M is 1
    input_dict = {
        "M": 1,
        "beta": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — larger beta value
    input_dict = {
        "M": 100,
        "beta": np.array(14.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — negative beta
    input_dict = {
        "M": 15,
        "beta": np.array(-2.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — standard typical Kaiser beta
    input_dict = {
        "M": 8,
        "beta": np.array(8.6, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — integer beta
    input_dict = {
        "M": 32,
        "beta": np.array(10, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — small positive beta
    input_dict = {
        "M": 50,
        "beta": np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — large M
    input_dict = {
        "M": 1000,
        "beta": np.array(6.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.kaiser_2"] = kaiser_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kaiser_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kaiser_2'.")


check_valid('jax.numpy.kaiser', generated_inputs['jax.numpy.kaiser_2'], lib="jax", suffix=2)
