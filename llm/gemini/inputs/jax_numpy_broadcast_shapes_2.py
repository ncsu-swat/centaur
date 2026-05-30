
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_shapes_inputs():
    list_of_inputs = []

    # Input 1: 1D shape with 1 element
    shapes = [1]
    list_of_inputs.append({"shapes": shapes})

    # Input 2: 1D shape with 4 elements
    shapes = [4]
    list_of_inputs.append({"shapes": shapes})

    # Input 3: 2D shape (3, 1)
    shapes = [3, 1]
    list_of_inputs.append({"shapes": shapes})

    # Input 4: 2D shape (1, 4)
    shapes = [1, 4]
    list_of_inputs.append({"shapes": shapes})

    # Input 5: 3D shape (5, 1, 1)
    shapes = [5, 1, 1]
    list_of_inputs.append({"shapes": shapes})

    # Input 6: 0D shape (scalar)
    shapes = []
    list_of_inputs.append({"shapes": shapes})

    # Input 7: 3D shape (2, 3, 4)
    shapes = [2, 3, 4]
    list_of_inputs.append({"shapes": shapes})

    # Input 8: 3D shape with larger dimensions
    shapes = [10, 10, 10]
    list_of_inputs.append({"shapes": shapes})

    # Input 9: 4D shape with ones
    shapes = [1, 1, 1, 1]
    list_of_inputs.append({"shapes": shapes})

    # Input 10: 2D square shape
    shapes = [5, 5]
    list_of_inputs.append({"shapes": shapes})

    # Input 11: 4D shape (1, 2, 3, 4)
    shapes = [1, 2, 3, 4]
    list_of_inputs.append({"shapes": shapes})

    return list_of_inputs


generated_inputs["jax.numpy.broadcast_shapes_2"] = broadcast_shapes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_shapes_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_shapes_2'.")


check_valid('jax.numpy.broadcast_shapes', generated_inputs['jax.numpy.broadcast_shapes_2'], lib="jax", suffix=2)
