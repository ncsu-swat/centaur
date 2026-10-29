
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divmod_inputs():
    list_of_inputs = []

    # Input 1: positive float and 1D float array
    x1 = 10.5
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: negative float and 1D positive float array
    x1 = -15.75
    x2 = np.array([2.5, 5.0, 7.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: positive float and 2D float array
    x1 = 100.0
    x2 = np.array([[3.0, 7.0], [11.0, 13.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: positive float and 2D negative float array
    x1 = 42.0
    x2 = np.array([[-5.0, -8.0], [-10.0, -2.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: negative float and negative float array
    x1 = -50.25
    x2 = np.array([-2.5, -4.0, -6.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: zero float and non-zero float array
    x1 = 0.0
    x2 = np.array([1.5, -3.2, 4.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: float and 3D array
    x1 = 25.5
    x2 = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: small float and 1D float array with small numbers
    x1 = 0.005
    x2 = np.array([0.001, 0.002, 0.003], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: float and 4D float array
    x1 = 12.0
    x2 = np.full((2, 2, 2, 2), 3.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: float and 0D float array (scalar array)
    x1 = 7.3
    x2 = np.array(2.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: float and integer tensor divisor
    x1 = 20.0
    x2 = np.array([[2, 3, 5], [7, 11, 13]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_5"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_5'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_5'], lib="jax", suffix=5)
