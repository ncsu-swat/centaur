
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def exp1_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 array
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of positive floats
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array float64
    x = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of floats
    x = np.random.uniform(0.5, 5.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array with small positive values close to 0
    x = np.array([[1e-4, 1e-3], [1e-2, 1e-1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array
    x = np.random.exponential(scale=2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array
    x = np.random.uniform(1.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (casted internally)
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with large values
    x = np.array([50.0, 100.0, 200.0, 500.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing infinity
    x = np.array([0.5, 1.0, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D integer array
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.exp1"] = exp1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.exp1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.exp1'.")


check_valid('jax.scipy.special.exp1', generated_inputs['jax.scipy.special.exp1'], lib="jax", suffix=0)
