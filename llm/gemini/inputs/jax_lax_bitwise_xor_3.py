
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays of the same shape
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: Python boolean scalars
    x = True
    y = False
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D boolean arrays of the same shape
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[False, True], [False, False]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D array and Python scalar (broadcasting)
    x = np.array([True, False, True], dtype=np.bool_)
    y = True
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 3D boolean arrays of the same shape
    x = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 2D array and Python scalar (broadcasting)
    x = np.array([[True, False], [True, True]], dtype=np.bool_)
    y = False
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Broadcasting with same number of dimensions (2D: (1, 2) and (2, 1))
    x = np.array([[True, False]], dtype=np.bool_)
    y = np.array([[False], [True]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 4D boolean arrays of the same shape
    x = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Multi-dimensional broadcasting with same number of dimensions (3D)
    x = np.random.choice([True, False], size=(1, 3, 1)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 1, 4)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 5D boolean arrays of the same shape
    x = np.random.choice([True, False], size=(2, 1, 2, 1, 2)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 1, 2, 1, 2)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.bitwise_xor_3"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_3'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_3'], lib="jax", suffix=3)
