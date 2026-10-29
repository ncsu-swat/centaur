
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def helmert_inputs():
    list_of_inputs = []

    # Input 1: n=1, full=True
    input_dict = {"n": 1, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: n=1, full=False
    input_dict = {"n": 1, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: n=2, full=True
    input_dict = {"n": 2, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: n=2, full=False
    input_dict = {"n": 2, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: n=3, full=True
    input_dict = {"n": 3, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: n=3, full=False
    input_dict = {"n": 3, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: n=4, full=True
    input_dict = {"n": 4, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: n=5, full=False
    input_dict = {"n": 5, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: n=8, full=True
    input_dict = {"n": 8, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: n=10, full=False
    input_dict = {"n": 10, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: n=16, full=True
    input_dict = {"n": 16, "full": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: n=32, full=False
    input_dict = {"n": 32, "full": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.helmert"] = helmert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.helmert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.helmert'.")


check_valid('jax.scipy.linalg.helmert', generated_inputs['jax.scipy.linalg.helmert'], lib="jax", suffix=0)
