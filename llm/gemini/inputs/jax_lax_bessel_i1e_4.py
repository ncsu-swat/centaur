
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    input_dict = {"x": float(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": float(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float32(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float32(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float64(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float64(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float32(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": np.float64(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_4"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_4'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_4'], lib="jax", suffix=4)
