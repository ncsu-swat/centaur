
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def sin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer 2
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer -2
    input_dict = {"x": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative integer
    input_dict = {"x": -90}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Angle-like integer 180
    input_dict = {"x": 180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative large integer
    input_dict = {"x": -360}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary integer
    input_dict = {"x": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sin_3"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_3'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_3'], lib="jax", suffix=3)
