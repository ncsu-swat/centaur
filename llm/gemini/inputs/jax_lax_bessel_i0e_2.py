
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def bessel_i0e_inputs():
    list_of_inputs = []

    values = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.71828,
        -3.14159,
        10.0,
        -10.0,
        50.5,
        -100.0,
        1e-4,
        -1e-4,
        1000.0,
    ]

    for val in values:
        input_dict = {"x": float(val)}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bessel_i0e_2"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_2'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_2'], lib="jax", suffix=2)
