
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: Zero to int32
    input_dict = {"operand": 0, "new_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer to float32
    input_dict = {"operand": 1, "new_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer to uint32
    input_dict = {"operand": -1, "new_dtype": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer to int16
    input_dict = {"operand": 42, "new_dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer to uint16
    input_dict = {"operand": -100, "new_dtype": np.dtype(np.uint16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer to int8
    input_dict = {"operand": 255, "new_dtype": np.dtype(np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative integer to uint8
    input_dict = {"operand": -128, "new_dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive integer to float16
    input_dict = {"operand": 1000000, "new_dtype": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative integer to int32
    input_dict = {"operand": -2147483648, "new_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Maximum positive 32-bit integer to uint32
    input_dict = {"operand": 2147483647, "new_dtype": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative integer to float32
    input_dict = {"operand": -999999, "new_dtype": np.dtype(np.float32)}
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
