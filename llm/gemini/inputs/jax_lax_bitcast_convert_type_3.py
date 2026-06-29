
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: float32 to int32 (same size)
    input_dict = {
        "operand": np.float32(1.0),
        "new_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 to uint32 (same size, negative value)
    input_dict = {
        "operand": np.float32(-1.0),
        "new_dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 to int64 (same size)
    input_dict = {
        "operand": np.float64(3.14159265),
        "new_dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 to uint64 (same size, negative value)
    input_dict = {
        "operand": np.float64(-3.14159265),
        "new_dtype": np.uint64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 to int16 (size decreases: 4 -> 2)
    input_dict = {
        "operand": np.float32(2.5),
        "new_dtype": np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 to uint16 (size decreases: 4 -> 2, negative value)
    input_dict = {
        "operand": np.float32(-2.5),
        "new_dtype": np.uint16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 to int32 (size decreases: 8 -> 4)
    input_dict = {
        "operand": np.float64(10.0),
        "new_dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 to uint32 (size decreases: 8 -> 4, negative value)
    input_dict = {
        "operand": np.float64(-10.0),
        "new_dtype": np.uint32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 to float32 (size decreases: 8 -> 4)
    input_dict = {
        "operand": np.float64(0.001),
        "new_dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 to int8 (size decreases: 4 -> 1)
    input_dict = {
        "operand": np.float32(0.0),
        "new_dtype": np.int8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 to uint8 (size decreases: 4 -> 1, negative zero)
    input_dict = {
        "operand": np.float32(-0.0),
        "new_dtype": np.uint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float64 to int16 (size decreases: 8 -> 2)
    input_dict = {
        "operand": np.float64(-123.456),
        "new_dtype": np.int16
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
