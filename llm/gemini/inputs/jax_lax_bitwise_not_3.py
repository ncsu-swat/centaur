
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: Scalar boolean
    list_of_inputs.append({"x": np.array(True, dtype=bool)})

    # Input 2: 1D boolean array
    list_of_inputs.append({"x": np.array([True, False, True, False], dtype=bool)})

    # Input 3: 2D boolean array
    list_of_inputs.append({"x": np.array([[True, False], [False, True]], dtype=bool)})

    # Input 4: 3D random boolean array
    list_of_inputs.append({"x": np.random.choice([True, False], size=(2, 3, 4)).astype(bool)})

    # Input 5: 4D boolean array with singleton dimensions
    list_of_inputs.append({"x": np.random.choice([True, False], size=(1, 5, 1, 5)).astype(bool)})

    # Input 6: All True boolean array
    list_of_inputs.append({"x": np.ones((10,), dtype=bool)})

    # Input 7: All False boolean array
    list_of_inputs.append({"x": np.zeros((8, 8), dtype=bool)})

    # Input 8: Large 2D boolean array
    list_of_inputs.append({"x": np.random.choice([True, False], size=(100, 100)).astype(bool)})

    # Input 9: 5D boolean array
    list_of_inputs.append({"x": np.random.choice([True, False], size=(2, 2, 2, 2, 2)).astype(bool)})

    # Input 10: Empty boolean array with shape
    list_of_inputs.append({"x": np.empty((0, 10), dtype=bool)})

    return list_of_inputs

generated_inputs["jax.lax.bitwise_not_3"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_3'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_3'], lib="jax", suffix=3)
