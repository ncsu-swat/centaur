
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: True to bool dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype(bool)})
    )

    # Input 2: False to bool dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype(bool)})
    )

    # Input 3: True to np.bool_ dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype(np.bool_)})
    )

    # Input 4: False to np.bool_ dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype(np.bool_)})
    )

    # Input 5: True to 'bool' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype("bool")})
    )

    # Input 6: False to 'bool' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype("bool")})
    )

    # Input 7: True to '?' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype("?")})
    )

    # Input 8: False to '?' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype("?")})
    )

    # Input 9: True to 'b1' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype("b1")})
    )

    # Input 10: False to 'b1' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype("b1")})
    )

    # Input 11: True to '|b1' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": True, "new_dtype": np.dtype("|b1")})
    )

    # Input 12: False to '|b1' dtype
    list_of_inputs.append(
        copy.deepcopy({"operand": False, "new_dtype": np.dtype("|b1")})
    )

    return list_of_inputs


generated_inputs["jax.lax.bitcast_convert_type_4"] = bitcast_convert_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitcast_convert_type_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitcast_convert_type_4'.")


check_valid('jax.lax.bitcast_convert_type', generated_inputs['jax.lax.bitcast_convert_type_4'], lib="jax", suffix=4)
