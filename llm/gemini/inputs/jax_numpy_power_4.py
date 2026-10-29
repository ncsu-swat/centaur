
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: positive integer base, 1D integer exponent array
    input_dict = {
        "x1": 2,
        "x2": np.array([0, 1, 2, 3, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive integer base, 2D integer exponent array
    input_dict = {
        "x1": 3,
        "x2": np.array([[1, 2], [3, 0]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative integer base, 1D integer exponent array
    input_dict = {
        "x1": -2,
        "x2": np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer base, 1D float32 exponent array
    input_dict = {
        "x1": 10,
        "x2": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: positive integer base, 2D fractional float exponent array
    input_dict = {
        "x1": 4,
        "x2": np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: zero base, 1D positive integer exponent array
    input_dict = {
        "x1": 0,
        "x2": np.array([1, 2, 3, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: base 1, 3D float64 exponent array
    input_dict = {
        "x1": 1,
        "x2": np.random.randn(2, 3, 4).astype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: positive integer base, 0D scalar tensor exponent
    input_dict = {
        "x1": 5,
        "x2": np.array(3, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: -1 base with alternating even/odd integer exponent array
    input_dict = {
        "x1": -1,
        "x2": np.arange(10, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: positive integer base, 4D integer exponent array
    input_dict = {
        "x1": 2,
        "x2": np.ones((2, 2, 2, 2), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: positive integer base, 1D float array with negative exponents
    input_dict = {
        "x1": 3,
        "x2": np.array([-1.0, -2.0, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: positive integer base, 2D int32 exponent array
    input_dict = {
        "x1": 7,
        "x2": np.zeros((3, 1), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.power_4"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_4'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_4'], lib="jax", suffix=4)
