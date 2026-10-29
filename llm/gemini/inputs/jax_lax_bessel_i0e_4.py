
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: float32 zero
    input_dict = {"x": np.float32(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 positive
    input_dict = {"x": np.float32(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 negative
    input_dict = {"x": np.float32(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 fractional positive
    input_dict = {"x": np.float32(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 large negative
    input_dict = {"x": np.float32(-5.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 large positive
    input_dict = {"x": np.float32(10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 small positive
    input_dict = {"x": np.float32(0.25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 zero
    input_dict = {"x": np.float64(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 positive
    input_dict = {"x": np.float64(1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 negative
    input_dict = {"x": np.float64(-3.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float64 large value
    input_dict = {"x": np.float64(50.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Python float positive
    input_dict = {"x": float(0.75)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Python float negative
    input_dict = {"x": float(-2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bessel_i0e_4"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_4'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_4'], lib="jax", suffix=4)
