
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def iterable_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"y": 0})

    # Input 2: Positive small integer
    list_of_inputs.append({"y": 1})

    # Input 3: Negative small integer
    list_of_inputs.append({"y": -1})

    # Input 4: Moderate positive integer
    list_of_inputs.append({"y": 42})

    # Input 5: Moderate negative integer
    list_of_inputs.append({"y": -100})

    # Input 6: Large positive integer
    list_of_inputs.append({"y": 1000000})

    # Input 7: Large negative integer
    list_of_inputs.append({"y": -999999})

    # Input 8: 32-bit signed int max
    list_of_inputs.append({"y": 2147483647})

    # Input 9: 32-bit signed int min
    list_of_inputs.append({"y": -2147483648})

    # Input 10: 64-bit boundary integer
    list_of_inputs.append({"y": 9223372036854775807})

    # Input 11: Negative 64-bit boundary integer
    list_of_inputs.append({"y": -9223372036854775808})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.iterable_4"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_4'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_4'], lib="jax", suffix=4)
