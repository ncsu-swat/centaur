
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def kaiser_inputs():
    list_of_inputs = []

    # Input 1: standard positive window size with float32 beta
    input_dict = {"M": 12, "beta": np.array(14.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: beta = 0.0 (rectangular window) with float64
    input_dict = {"M": 5, "beta": np.array(0.0, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: M = 1 boundary case
    input_dict = {"M": 1, "beta": np.array(5.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: small window size M = 2
    input_dict = {"M": 2, "beta": np.array(2.5, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small window size M = 3
    input_dict = {"M": 3, "beta": np.array(1.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: larger window size with typical Kaiser beta value
    input_dict = {"M": 64, "beta": np.array(8.6, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative beta
    input_dict = {"M": 10, "beta": np.array(-3.5, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: integer dtype beta tensor
    input_dict = {"M": 16, "beta": np.array(6, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 beta with odd M
    input_dict = {"M": 25, "beta": np.array(4.2, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large beta
    input_dict = {"M": 32, "beta": np.array(20.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 dtype beta tensor
    input_dict = {"M": 20, "beta": np.array(1.5, dtype=np.float16)}
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
