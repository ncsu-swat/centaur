
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: positive float to int32
    input_dict = {"operand": 3.14, "new_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float to int32
    input_dict = {"operand": -5.99, "new_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero float to bool
    input_dict = {"operand": 0.0, "new_dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: non-zero float to bool
    input_dict = {"operand": 1.5, "new_dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float to float32
    input_dict = {"operand": 2.718281828, "new_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float to float64
    input_dict = {"operand": -123.456, "new_dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float to uint8
    input_dict = {"operand": 255.4, "new_dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float to int16
    input_dict = {"operand": -1024.5, "new_dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small float to int8
    input_dict = {"operand": 12.0, "new_dtype": np.dtype(np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large float to int64
    input_dict = {"operand": 1e9, "new_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float to complex64
    input_dict = {"operand": -42.0, "new_dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float to uint32
    input_dict = {"operand": 65536.8, "new_dtype": np.dtype(np.uint32)}
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
