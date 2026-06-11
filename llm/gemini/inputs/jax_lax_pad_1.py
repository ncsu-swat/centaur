
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive low and high padding, int32
    operand = np.array([1, 2, 3], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = [(1, 2, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, interior padding (dilation), int32
    operand = np.array([1, 2, 3], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = [(0, 0, 1)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, low and high padding in both dimensions, float32
    operand = np.random.randn(2, 3).astype(np.float32)
    padding_value = np.array(-1.0, dtype=np.float32)
    padding_config = [(1, 1, 0), (2, 2, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, negative padding (edge removal), float32
    operand = np.random.randn(5, 5).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = [(-1, -1, 0), (-2, -1, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, negative padding, int32
    operand = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    padding_value = np.array(9, dtype=np.int32)
    padding_config = [(-1, -2, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, mixed padding, float64
    operand = np.random.randn(2, 2, 2).astype(np.float64)
    padding_value = np.array(0.0, dtype=np.float64)
    padding_config = [(1, 1, 1), (0, 0, 0), (0, 1, 2)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, only interior padding, int64
    operand = np.ones((3, 3), dtype=np.int64)
    padding_value = np.array(-9, dtype=np.int64)
    padding_config = [(0, 0, 2), (0, 0, 1)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, simple padding, float32
    operand = np.random.randn(1, 2, 1, 2).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = [(0, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, padding with positive integer, int32
    operand = np.zeros((4,), dtype=np.int32)
    padding_value = np.array(5, dtype=np.int32)
    padding_config = [(2, 3, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, combined negative, positive and interior padding, float32
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    padding_value = np.array(-1.5, dtype=np.float32)
    padding_config = [(-1, -1, 0), (1, 2, 1), (0, -2, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D boolean array padding, bool
    operand = np.array([[True, False], [False, True]], dtype=bool)
    padding_value = np.array(True, dtype=bool)
    padding_config = [(1, 1, 0), (0, 1, 0)]
    input_dict = {
        "operand": operand,
        "padding_value": padding_value,
        "padding_config": padding_config
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.pad_1"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pad_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pad_1'.")


check_valid('jax.lax.pad', generated_inputs['jax.lax.pad_1'], lib="jax", suffix=1)
