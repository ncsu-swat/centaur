
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: List of integers
    input_dict = {"x": [1, 2, 3, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of negative integers
    input_dict = {"x": [-10, -20, -30]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of floats
    input_dict = {"x": [1.5, -2.3, 0.0, 4.7]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of complex numbers
    input_dict = {"x": [1 + 2j, 3 - 4j, -5j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed list containing real and complex numbers
    input_dict = {"x": [0, 1.5, -2 + 3j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of booleans
    input_dict = {"x": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Nested 2D list of integers
    input_dict = {"x": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Nested 2D list of complex numbers
    input_dict = {"x": [[1 + 0j, 2 - 1j], [-3j, 4.0 + 0j]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Nested 3D list of floats
    input_dict = {"x": [[[1.0, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested 3D list with complex elements
    input_dict = {"x": [[[0j, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0 + 1j]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty list
    input_dict = {"x": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
