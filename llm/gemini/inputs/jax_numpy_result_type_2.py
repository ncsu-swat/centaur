
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: int32
    list_of_inputs.append({"args": "int32"})

    # Input 2: float32
    list_of_inputs.append({"args": "float32"})

    # Input 3: int64
    list_of_inputs.append({"args": "int64"})

    # Input 4: float64
    list_of_inputs.append({"args": "float64"})

    # Input 5: bool
    list_of_inputs.append({"args": "bool"})

    # Input 6: uint8
    list_of_inputs.append({"args": "uint8"})

    # Input 7: int16
    list_of_inputs.append({"args": "int16"})

    # Input 8: float16
    list_of_inputs.append({"args": "float16"})

    # Input 9: complex64
    list_of_inputs.append({"args": "complex64"})

    # Input 10: uint32
    list_of_inputs.append({"args": "uint32"})

    # Input 11: int8
    list_of_inputs.append({"args": "int8"})

    # Input 12: complex128
    list_of_inputs.append({"args": "complex128"})

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
