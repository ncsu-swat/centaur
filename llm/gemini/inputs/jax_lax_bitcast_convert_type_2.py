
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: int32(0) to float32
    input_dict = {
        "operand": np.int32(0),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32(1) to float32
    input_dict = {
        "operand": np.int32(1),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32(-1) to float32
    input_dict = {
        "operand": np.int32(-1),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32(100) to int32
    input_dict = {
        "operand": np.int32(100),
        "new_dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32(-100) to int32
    input_dict = {
        "operand": np.int32(-100),
        "new_dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32(123456789) to float32
    input_dict = {
        "operand": np.int32(123456789),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32(-123456789) to float32
    input_dict = {
        "operand": np.int32(-123456789),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32(2147483647) to float32
    input_dict = {
        "operand": np.int32(2147483647),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int32(-2147483648) to float32
    input_dict = {
        "operand": np.int32(-2147483648),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32(42) to int32
    input_dict = {
        "operand": np.int32(42),
        "new_dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int32(1337) to float32
    input_dict = {
        "operand": np.int32(1337),
        "new_dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bitcast_convert_type_2"] = bitcast_convert_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitcast_convert_type_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitcast_convert_type_2'.")


check_valid('jax.lax.bitcast_convert_type', generated_inputs['jax.lax.bitcast_convert_type_2'], lib="jax", suffix=2)
