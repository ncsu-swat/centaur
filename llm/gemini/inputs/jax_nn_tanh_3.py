
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tanh_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0})

    # Input 2: Positive small integer
    list_of_inputs.append({"x": 1})

    # Input 3: Negative small integer
    list_of_inputs.append({"x": -1})

    # Input 4: Positive medium integer
    list_of_inputs.append({"x": 5})

    # Input 5: Negative medium integer
    list_of_inputs.append({"x": -5})

    # Input 6: Large positive integer
    list_of_inputs.append({"x": 100})

    # Input 7: Large negative integer
    list_of_inputs.append({"x": -100})

    # Input 8: numpy int32 positive
    list_of_inputs.append({"x": np.int32(10)})

    # Input 9: numpy int64 negative
    list_of_inputs.append({"x": np.int64(-10)})

    # Input 10: numpy int16 positive
    list_of_inputs.append({"x": np.int16(2)})

    return list_of_inputs


generated_inputs["jax.nn.tanh_3"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.tanh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.tanh_3'.")


check_valid('jax.nn.tanh', generated_inputs['jax.nn.tanh_3'], lib="jax", suffix=3)
