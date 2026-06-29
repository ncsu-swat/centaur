
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_bitcast_convert_type_inputs():
    list_of_inputs = []

    # Input 1: scalar bool
    operand = np.array(True, dtype=bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 2: 1D bool array
    operand = np.array([True, False], dtype=bool)
    new_dtype = np.dtype(bool)
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 3: 2D bool array
    operand = np.array([[True, False], [False, True]], dtype=bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 4: 3D bool array
    operand = np.array([[[True, True], [False, False]]], dtype=bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 5: 1D bool array larger size
    operand = np.random.choice([True, False], size=(10,)).astype(bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 6: 3D bool array (3, 3, 3)
    operand = np.random.choice([True, False], size=(3, 3, 3)).astype(bool)
    new_dtype = np.dtype(bool)
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 7: 2D bool array (2, 5)
    operand = np.random.choice([True, False], size=(2, 5)).astype(bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 8: 4D bool array (1, 1, 1, 5)
    operand = np.random.choice([True, False], size=(1, 1, 1, 5)).astype(bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 9: 1D bool array of size 1
    operand = np.array([False], dtype=bool)
    new_dtype = np.bool_
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    # Input 10: 3D bool array (4, 2, 2)
    operand = np.random.choice([True, False], size=(4, 2, 2)).astype(bool)
    new_dtype = np.dtype(bool)
    list_of_inputs.append({"operand": operand, "new_dtype": new_dtype})

    return list_of_inputs

generated_inputs["jax.lax.bitcast_convert_type_4"] = jax_lax_bitcast_convert_type_inputs()

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
