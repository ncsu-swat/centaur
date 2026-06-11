
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: 0D boolean, dtype float32
    operand = np.array(True, dtype=bool)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 2: 1D boolean, dtype int32
    operand = np.array([True, False, True], dtype=bool)
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 3: 2D boolean, dtype float64
    operand = np.array([[True, False], [False, True]], dtype=bool)
    new_dtype = np.dtype(np.float64)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 4: 3D boolean, dtype int64
    operand = np.random.choice([True, False], size=(2, 3, 4)).astype(bool)
    new_dtype = np.dtype(np.int64)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 5: 4D boolean, dtype uint8
    operand = np.random.choice([True, False], size=(1, 2, 2, 3)).astype(bool)
    new_dtype = np.dtype(np.uint8)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 6: 1D boolean, dtype int16
    operand = np.array([False, False, True, True], dtype=bool)
    new_dtype = np.dtype(np.int16)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 7: 2D boolean, dtype float16
    operand = np.random.choice([True, False], size=(5, 5)).astype(bool)
    new_dtype = np.dtype(np.float16)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 8: 3D boolean, dtype complex64
    operand = np.random.choice([True, False], size=(2, 2, 2)).astype(bool)
    new_dtype = np.dtype(np.complex64)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 9: 1D boolean, dtype bool
    operand = np.array([True, True], dtype=bool)
    new_dtype = np.dtype(bool)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    # Input 10: 5D boolean, dtype int8
    operand = np.random.choice([True, False], size=(1, 1, 2, 2, 2)).astype(bool)
    new_dtype = np.dtype(np.int8)
    list_of_inputs.append({"operand": copy.deepcopy(operand), "new_dtype": new_dtype})

    return list_of_inputs

generated_inputs["jax.lax.convert_element_type_4"] = convert_element_type_inputs()

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
