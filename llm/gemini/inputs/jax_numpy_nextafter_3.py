
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: positive towards smaller positive
    input_dict = {"x": 2, "y": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: zero towards positive
    input_dict = {"x": 0, "y": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero towards negative
    input_dict = {"x": 0, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative towards smaller magnitude negative
    input_dict = {"x": -2, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: positive towards larger positive
    input_dict = {"x": 1, "y": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: identical values
    input_dict = {"x": 10, "y": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative towards positive
    input_dict = {"x": -5, "y": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large positive towards zero
    input_dict = {"x": 1000, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative towards larger magnitude negative
    input_dict = {"x": -100, "y": -200}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: positive towards negative
    input_dict = {"x": 42, "y": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: both zero
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nextafter_3"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nextafter_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nextafter_3'.")


check_valid('jax.numpy.nextafter', generated_inputs['jax.numpy.nextafter_3'], lib="jax", suffix=3)
