
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: True to float32
    input_dict = {"operand": True, "new_dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: False to int32
    input_dict = {"operand": False, "new_dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: True to bool
    input_dict = {"operand": True, "new_dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: False to float64
    input_dict = {"operand": False, "new_dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: True to uint8
    input_dict = {"operand": True, "new_dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: False to int8
    input_dict = {"operand": False, "new_dtype": np.dtype(np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: True to int16
    input_dict = {"operand": True, "new_dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: False to uint32
    input_dict = {"operand": False, "new_dtype": np.dtype(np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: True to float16
    input_dict = {"operand": True, "new_dtype": np.dtype(np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: False to complex64
    input_dict = {"operand": False, "new_dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: True to int64
    input_dict = {"operand": True, "new_dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.convert_element_type_4"] = (
    convert_element_type_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.convert_element_type_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.convert_element_type_4'.")


check_valid('jax.lax.convert_element_type', generated_inputs['jax.lax.convert_element_type_4'], lib="jax", suffix=4)
