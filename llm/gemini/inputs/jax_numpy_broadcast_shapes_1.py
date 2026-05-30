
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_shapes_inputs():
    list_of_inputs = []

    # Input 1: 1D shape
    input_dict = {"shapes": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: another 1D shape
    input_dict = {"shapes": (4,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D shape
    input_dict = {"shapes": (3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape
    input_dict = {"shapes": (5, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D shape (scalar)
    input_dict = {"shapes": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D square shape
    input_dict = {"shapes": (5, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D shape
    input_dict = {"shapes": (2, 3, 4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D large shape
    input_dict = {"shapes": (10, 20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: another 4D shape
    input_dict = {"shapes": (8, 7, 6, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D unit shape
    input_dict = {"shapes": (1, 1, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D larger shape
    input_dict = {"shapes": (100, 200, 300)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_shapes_1"] = broadcast_shapes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_shapes_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_shapes_1'.")


check_valid('jax.numpy.broadcast_shapes', generated_inputs['jax.numpy.broadcast_shapes_1'], lib="jax", suffix=1)
