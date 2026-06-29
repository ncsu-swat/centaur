
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def norm_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard normal (float)
    input_dict = {
        "x": float(0.0),
        "loc": float(0.0),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive offset
    input_dict = {
        "x": float(1.5),
        "loc": float(0.5),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x
    input_dict = {
        "x": float(-2.0),
        "loc": float(0.0),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large scale
    input_dict = {
        "x": float(5.0),
        "loc": float(0.0),
        "scale": float(10.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small scale
    input_dict = {
        "x": float(0.1),
        "loc": float(0.0),
        "scale": float(0.1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-zero mean, small scale
    input_dict = {
        "x": float(-0.5),
        "loc": float(-1.0),
        "scale": float(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.float64 type
    input_dict = {
        "x": np.float64(1.25),
        "loc": np.float64(0.0),
        "scale": np.float64(2.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.float32 type
    input_dict = {
        "x": np.float32(-0.75),
        "loc": np.float32(0.25),
        "scale": np.float32(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    input_dict = {
        "x": float(100.0),
        "loc": float(50.0),
        "scale": float(25.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Near-zero inputs
    input_dict = {
        "x": float(1e-5),
        "loc": float(-1e-5),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.pdf_2"] = norm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.pdf_2'.")


check_valid('jax.scipy.stats.norm.pdf', generated_inputs['jax.scipy.stats.norm.pdf_2'], lib="jax", suffix=2)
