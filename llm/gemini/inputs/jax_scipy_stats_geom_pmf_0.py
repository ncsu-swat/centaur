
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def geom_pmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard case
    input_dict = {
        "k": np.array(3, dtype=np.int32),
        "p": np.array(0.5, dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of matching shape
    input_dict = {
        "k": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "p": np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "loc": np.array([0, 0, 0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-zero location parameter
    input_dict = {
        "k": np.array([3, 4, 5], dtype=np.float32),
        "p": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([2, 2, 2], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative location parameter
    input_dict = {
        "k": np.array([0, 1, 2], dtype=np.int32),
        "p": np.array([0.3, 0.3, 0.3], dtype=np.float32),
        "loc": np.array([-2, -2, -2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays with broadcasting
    input_dict = {
        "k": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "p": np.array([[0.25], [0.75]], dtype=np.float32),
        "loc": np.array([[0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 precision
    input_dict = {
        "k": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "p": np.array([0.1, 0.5, 0.9], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Values outside support (k < loc + 1)
    input_dict = {
        "k": np.array([-1, 0, 1], dtype=np.int32),
        "p": np.array([0.4, 0.4, 0.4], dtype=np.float32),
        "loc": np.array([0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {
        "k": np.ones((2, 3, 4), dtype=np.int32) * 2,
        "p": np.full((2, 3, 4), 0.6, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme probability p = 1.0
    input_dict = {
        "k": np.array([1, 2, 3], dtype=np.int32),
        "p": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "loc": np.array([0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting 1D k with 2D p and scalar loc
    input_dict = {
        "k": np.array([1, 2, 3, 4], dtype=np.int32),
        "p": np.array([[0.1], [0.5], [0.9]], dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.geom.pmf"] = geom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.geom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.geom.pmf'.")


check_valid('jax.scipy.stats.geom.pmf', generated_inputs['jax.scipy.stats.geom.pmf'], lib="jax", suffix=0)
