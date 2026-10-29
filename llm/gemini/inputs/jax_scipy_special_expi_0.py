
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expi_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    x = np.array([0.1, 1.0, 2.5, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of negative floats
    x = np.array([-0.1, -1.0, -2.5, -10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float64)
    x = np.array(1.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of mixed positive and negative values (float64)
    x = np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array with small positive values (float32)
    x = np.linspace(0.01, 1.0, 8).reshape(2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Large positive and negative values (float64)
    x = np.array([-50.0, -20.0, 20.0, 50.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array (float32)
    x = np.ones((2, 1, 3, 2), dtype=np.float32) * 0.75
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Integer values cast to float32
    x = np.array([-5, -1, 1, 5, 10], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing special values like inf and -inf
    x = np.array([-np.inf, -100.0, 0.001, np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D array with random normal values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.expi"] = expi_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.expi' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.expi'.")


check_valid('jax.scipy.special.expi', generated_inputs['jax.scipy.special.expi'], lib="jax", suffix=0)
