
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1
    operand = np.array([1, 2], dtype=np.int32)
    new_sizes = [2, 1]
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 2
    operand = np.array([[1, 2]], dtype=np.float32)
    new_sizes = [2]
    dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 3
    operand = np.array([[1], [2]], dtype=np.float32)
    new_sizes = [2]
    dimensions = (1, 0)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 4
    operand = np.array([[[1, 2]]], dtype=np.int32)
    new_sizes = [2, 1]
    dimensions = (0, 1, 2)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 5
    operand = np.array([[1, 2], [3, 4]], dtype=np.float32)
    new_sizes = [4]
    dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 6
    operand = np.array([[1, 2], [3, 4]], dtype=np.float32)
    new_sizes = [4]
    dimensions = (1, 0)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 7
    operand = np.array([1, 2, 3, 4], dtype=np.int32)
    new_sizes = [2, 2]
    dimensions = (0,)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 8
    operand = np.array([[[1], [2]]], dtype=np.float32)
    new_sizes = [2]
    dimensions = (0, 1, 2)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 9
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    new_sizes = [6]
    dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    # Input 10
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    new_sizes = [3, 2]
    dimensions = (1, 0)
    list_of_inputs.append({"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions})

    return list_of_inputs

generated_inputs["jax.lax.reshape_3"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_3'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_3'], lib="jax", suffix=3)
