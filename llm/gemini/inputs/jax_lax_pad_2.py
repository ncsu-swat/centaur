
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array, simple low/high padding
    operand = np.array([1, 2, 3], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = ((1, 2, 0),)
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, interior padding (dilation)
    operand = np.array([1.0, 2.0], dtype=np.float32)
    padding_value = np.array(-1.0, dtype=np.float32)
    padding_config = ((0, 0, 2),)
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, negative edge padding (cropping)
    operand = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = ((-1, -1, 0),)
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, standard padding on both dimensions
    operand = np.random.randn(3, 4).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = ((1, 1, 0), (2, 2, 0))
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, interior padding on both dimensions
    operand = np.random.randn(2, 2).astype(np.float64)
    padding_value = np.array(0.0, dtype=np.float64)
    padding_config = ((0, 0, 1), (0, 0, 1))
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, mixed positive and negative padding
    operand = np.random.randint(0, 10, size=(4, 4)).astype(np.int32)
    padding_value = np.array(-1, dtype=np.int32)
    padding_config = ((-1, 2, 0), (2, -1, 0))
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, complex padding configuration
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    padding_value = np.array(9.9, dtype=np.float32)
    padding_config = ((1, 0, 0), (0, 1, 1), (1, 1, 0))
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, uniform padding
    operand = np.zeros((2, 2, 2, 2), dtype=np.float32)
    padding_value = np.array(1.0, dtype=np.float32)
    padding_config = ((1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0))
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D array (scalar array)
    operand = np.array(5.0, dtype=np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = ()
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, boolean type
    operand = np.array([True, False, True], dtype=np.bool_)
    padding_value = np.array(False, dtype=np.bool_)
    padding_config = ((1, 1, 1),)
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array, int64 type
    operand = np.array([100, 200], dtype=np.int64)
    padding_value = np.array(-999, dtype=np.int64)
    padding_config = ((2, 3, 0),)
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.pad_2"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pad_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pad_2'.")


check_valid('jax.lax.pad', generated_inputs['jax.lax.pad_2'], lib="jax", suffix=2)
