
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: positive float to int32
    input_dict = {
        "operand": float(1.0),
        "new_dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float to uint32
    input_dict = {
        "operand": float(-2.5),
        "new_dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero float to float32
    input_dict = {
        "operand": float(0.0),
        "new_dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative zero float to int16
    input_dict = {
        "operand": float(-0.0),
        "new_dtype": np.dtype(np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: fractional float to uint16
    input_dict = {
        "operand": float(3.14159265),
        "new_dtype": np.dtype(np.uint16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large magnitude negative float to float16
    input_dict = {
        "operand": float(-100.25),
        "new_dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: small positive float to int8
    input_dict = {
        "operand": float(1e-4),
        "new_dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large power of 2 float to uint8
    input_dict = {
        "operand": float(65536.0),
        "new_dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: positive infinity to int32
    input_dict = {
        "operand": float("inf"),
        "new_dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative infinity to uint32
    input_dict = {
        "operand": float("-inf"),
        "new_dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN to int16
    input_dict = {
        "operand": float("nan"),
        "new_dtype": np.dtype(np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: large negative scientific notation float to int8
    input_dict = {
        "operand": float(-1e10),
        "new_dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.bitcast_convert_type_3"] = bitcast_convert_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitcast_convert_type_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitcast_convert_type_3'.")


check_valid('jax.lax.bitcast_convert_type', generated_inputs['jax.lax.bitcast_convert_type_3'], lib="jax", suffix=3)
