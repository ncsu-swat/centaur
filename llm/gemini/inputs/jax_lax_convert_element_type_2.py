
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: 0-D integer scalar, to float32
    input_dict = {
        "operand": np.array(42, dtype=np.int32),
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D integer array with negative values, to float64
    input_dict = {
        "operand": np.array([-10, 0, 10, 20], dtype=np.int32),
        "new_dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D int16 array, to int32
    input_dict = {
        "operand": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "new_dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3-D int8 array, to float32
    input_dict = {
        "operand": np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int8),
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1-D int64 array, to int8
    input_dict = {
        "operand": np.array([100, 200, 300], dtype=np.int64),
        "new_dtype": np.dtype(np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2-D uint32 array, to float16
    input_dict = {
        "operand": np.array([[10, 20], [30, 40]], dtype=np.uint32),
        "new_dtype": np.dtype(np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1-D int32 array, to bool
    input_dict = {
        "operand": np.array([0, 1, 2, 0], dtype=np.int32),
        "new_dtype": np.dtype(np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar int64, to float32
    input_dict = {
        "operand": np.int64(-999),
        "new_dtype": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4-D uint8 array, to int32
    input_dict = {
        "operand": np.ones((2, 2, 2, 2), dtype=np.uint8) * 255,
        "new_dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1-D int32 array (large values), to float64
    input_dict = {
        "operand": np.array([2147483647, -2147483648], dtype=np.int32),
        "new_dtype": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2-D int16 array with large dimensions, to uint32
    input_dict = {
        "operand": np.zeros((10, 10), dtype=np.int16),
        "new_dtype": np.dtype(np.uint32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.convert_element_type_2"] = convert_element_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.convert_element_type_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.convert_element_type_2'.")


check_valid('jax.lax.convert_element_type', generated_inputs['jax.lax.convert_element_type_2'], lib="jax", suffix=2)
