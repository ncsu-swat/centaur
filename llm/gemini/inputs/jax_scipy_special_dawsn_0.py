
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dawsn_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of positive values
    x = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array of negative values
    x = np.array([-0.1, -0.5, -1.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array spanning negative to positive
    x = np.linspace(-3.0, 3.0, 10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float32
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array float64
    x = np.array([[0.1, -0.2], [-0.3, 0.4]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array float32
    x = np.ones((2, 3, 4), dtype=np.float32) * 0.5
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 3D array with varying signs
    x = np.array([[[-1.0, 1.0], [2.0, -2.0]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D array float32
    x = np.zeros((1, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array of float64 values
    x = np.array([5.0, 10.0, 20.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.dawsn"] = dawsn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.dawsn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.dawsn'.")


check_valid('jax.scipy.special.dawsn', generated_inputs['jax.scipy.special.dawsn'], lib="jax", suffix=0)
