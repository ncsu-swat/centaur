
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: List of real integers
    list_of_inputs.append({"x": [1, 2, 3, 4]})

    # Input 2: List of complex numbers
    list_of_inputs.append({"x": [1+2j, -3-4j]})

    # Input 3: List of real floats
    list_of_inputs.append({"x": [1.0, -2.5, 3.14]})

    # Input 4: List containing a mix of real and complex numbers
    list_of_inputs.append({"x": [1.0, 2j, -3.5]})

    # Input 5: Empty list
    list_of_inputs.append({"x": []})

    # Input 6: 2D list of complex numbers
    list_of_inputs.append({"x": [[1+1j, 2j], [3, 4-2j]]})

    # Input 7: 2D list of real integers
    list_of_inputs.append({"x": [[1, -2], [3, 4]]})

    # Input 8: List of complex numbers with zero imaginary parts
    list_of_inputs.append({"x": [1+0j, -2.5+0j]})

    # Input 9: 3D nested list of complex numbers
    list_of_inputs.append({"x": [[[1j, -2j]], [[3j, 4j]]]})

    # Input 10: List of boolean values
    list_of_inputs.append({"x": [True, False, True]})

    # Input 11: List of numpy complex types
    list_of_inputs.append({"x": [np.complex64(1+1j), np.complex64(-2)]})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_5"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_5'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_5'], lib="jax", suffix=5)
