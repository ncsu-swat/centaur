
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def entr_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (positive float)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with standard probabilities
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array containing zeros and ones
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array containing negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array of uniform positive floats
    x = np.random.uniform(0.01, 1.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array with float64 values (mix of positive and negative)
    x = np.random.randn(2, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array of positive values
    x = np.random.exponential(scale=2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array of all zeros
    x = np.zeros((2, 2, 3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D integer array
    x = np.array([0, 1, 2, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with very small positive numbers
    x = np.array([1e-10, 1e-15, 1e-20, 1e-30], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with large values
    x = np.array([1e3, 1e5, 1e8], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.entr"] = entr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.entr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.entr'.")


check_valid('jax.scipy.special.entr', generated_inputs['jax.scipy.special.entr'], lib="jax", suffix=0)
