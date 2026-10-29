
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: 0-index for a 2D shape
    list_of_inputs.append({"indices": 0, "shape": (2, 3)})

    # Input 2: Mid-range index for a 2D shape
    list_of_inputs.append({"indices": 4, "shape": (3, 3)})

    # Input 3: Negative index for a 2D shape
    list_of_inputs.append({"indices": -1, "shape": (4, 5)})

    # Input 4: 1D shape with positive index
    list_of_inputs.append({"indices": 5, "shape": (10,)})

    # Input 5: 3D shape with scalar index
    list_of_inputs.append({"indices": 11, "shape": (2, 3, 4)})

    # Input 6: 4D shape
    list_of_inputs.append({"indices": 27, "shape": (2, 2, 3, 3)})

    # Input 7: Out-of-bounds positive index (clipped by JAX)
    list_of_inputs.append({"indices": 100, "shape": (3, 4)})

    # Input 8: Negative index with 3D shape
    list_of_inputs.append({"indices": -5, "shape": (3, 3, 3)})

    # Input 9: Single-element shape
    list_of_inputs.append({"indices": 0, "shape": (1, 1, 1)})

    # Input 10: 5D shape
    list_of_inputs.append({"indices": 42, "shape": (2, 2, 2, 2, 4)})

    # Input 11: Large index in higher-dimensional tensor
    list_of_inputs.append({"indices": 1234, "shape": (10, 10, 10, 10)})

    return list_of_inputs


generated_inputs["jax.numpy.unravel_index_3"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unravel_index_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unravel_index_3'.")


check_valid('jax.numpy.unravel_index', generated_inputs['jax.numpy.unravel_index_3'], lib="jax", suffix=3)
