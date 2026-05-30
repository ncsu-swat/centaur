
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: List of positive integers
    list_of_inputs.append({"y": [1, 2, 3, 4, 5]})

    # Input 2: List of negative integers
    list_of_inputs.append({"y": [-10, -5, -1]})

    # Input 3: List of mixed sign integers
    list_of_inputs.append({"y": [-5, 0, 5]})

    # Input 4: List of floats
    list_of_inputs.append({"y": [1.1, 2.2, 3.3]})

    # Input 5: List of negative floats
    list_of_inputs.append({"y": [-1.1, -2.2, -3.3]})

    # Input 6: List of mixed sign floats
    list_of_inputs.append({"y": [-1.5, 0.0, 1.5]})

    # Input 7: List of booleans
    list_of_inputs.append({"y": [True, False, True]})

    # Input 8: List with a single integer
    list_of_inputs.append({"y": [42]})

    # Input 9: List with a single float
    list_of_inputs.append({"y": [-0.001]})

    # Input 10: List of larger integers
    list_of_inputs.append({"y": [100, 200, 300, 400]})

    # Input 11: List of mixed integers and floats
    list_of_inputs.append({"y": [1, 2.5, -3, 4.2]})

    return list_of_inputs

generated_inputs["jax.numpy.iterable_2"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_2'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_2'], lib="jax", suffix=2)
