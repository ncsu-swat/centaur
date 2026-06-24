
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_polygamma_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "n": 0,
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "n": 1,
        "x": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "n": 2,
        "x": np.random.uniform(0.1, 10.0, size=(3, 3, 3)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "n": 3,
        "x": np.array([0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "n": 0,
        "x": np.random.uniform(1.0, 5.0, size=(5,)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "n": 1,
        "x": np.random.uniform(0.5, 2.5, size=(2, 4)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "n": 2,
        "x": np.array([10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "n": 4,
        "x": np.random.uniform(1.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "n": 0,
        "x": np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "n": 5,
        "x": np.random.uniform(5.0, 10.0, size=(1,)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.polygamma_1"] = jax_scipy_special_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_1'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_1'], lib="jax", suffix=1)
