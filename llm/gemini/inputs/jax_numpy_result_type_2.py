
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp


def result_type_inputs():
    list_of_inputs = []

    dtype_strings = [
        "int32",
        "float32",
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int64",
        "float64",
        "float16",
        "bfloat16",
        "complex64",
        "complex128",
        "uint32",
        "uint64",
    ]

    for dt_str in dtype_strings:
        input_dict = {"args": dt_str}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.result_type_2"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_2'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_2'], lib="jax", suffix=2)
