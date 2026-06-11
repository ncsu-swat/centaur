
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: positive float to int32
    input_dict = {
        "operand": 3.14,
        "new_dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float to int32
    input_dict = {
        "operand": -5.9,
        "new_dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero float to bool
    input_dict = {
        "operand": 0.0,
        "new_dtype": np.dtype(np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float to float32
    input_dict = {
        "operand": 1.234567,
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float to float64
    input_dict = {
        "operand": -987.654321,
        "new_dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive infinity to float32
    input_dict = {
        "operand": float('inf'),
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NaN to float32
    input_dict = {
        "operand": float('nan'),
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float to uint8
    input_dict = {
        "operand": 255.1,
        "new_dtype": np.dtype(np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative float to uint16
    input_dict = {
        "operand": -1.0,
        "new_dtype": np.dtype(np.uint16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float to complex64
    input_dict = {
        "operand": 2.71828,
        "new_dtype": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.convert_element_type_3"] = convert_element_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.convert_element_type_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.convert_element_type_3'.")


check_valid('jax.lax.convert_element_type', generated_inputs['jax.lax.convert_element_type_3'], lib="jax", suffix=3)
