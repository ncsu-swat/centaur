
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def igammac_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats, float32
    input_dict = {
        "a": np.float32(1.0),
        "x": np.float32(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-integer floats, float32
    input_dict = {
        "a": np.float32(0.5),
        "x": np.float32(1.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger a, smaller x, float64
    input_dict = {
        "a": np.float64(5.0),
        "x": np.float64(2.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small values, float32
    input_dict = {
        "a": np.float32(0.1),
        "x": np.float32(0.1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x is zero, float64
    input_dict = {
        "a": np.float64(2.5),
        "x": np.float64(0.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values, float64
    input_dict = {
        "a": np.float64(10.0),
        "x": np.float64(15.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small a, float32
    input_dict = {
        "a": np.float32(0.01),
        "x": np.float32(2.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High precision large float64
    input_dict = {
        "a": np.float64(50.0),
        "x": np.float64(50.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large x, float32
    input_dict = {
        "a": np.float32(1.5),
        "x": np.float32(100.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Typical distribution parameters, float64
    input_dict = {
        "a": np.float64(3.3),
        "x": np.float64(4.4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.igammac_2"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igammac_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igammac_2'.")


check_valid('jax.lax.igammac', generated_inputs['jax.lax.igammac_2'], lib="jax", suffix=2)
