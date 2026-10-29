
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default dimension
    operand = np.array([3.0, -1.0, 2.5, 0.0, -4.2], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, unstable sort along dimension 0
    operand = np.array([10, -5, 0, 7, -2, 100, -50], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": False,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, sort along dimension 0
    operand = np.array(
        [[1.5, -2.3, 0.0], [4.1, 1.1, -3.2], [-0.5, 2.2, 5.0]],
        dtype=np.float64,
    )
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, sort along dimension 1
    operand = np.random.randn(5, 8).astype(np.float32)
    input_dict = {
        "operand": operand,
        "dimension": 1,
        "is_stable": False,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, sort along last dimension
    operand = np.random.randint(-100, 100, size=(3, 4, 5), dtype=np.int64)
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, sort along middle dimension
    operand = np.random.randn(2, 6, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "dimension": 1,
        "is_stable": False,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, sort along dimension 2
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "dimension": 2,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array with special values (NaN, inf, signed zero)
    operand = np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, 1.0], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array, sort along last dimension
    operand = np.array(
        [[1.0 + 2.0j, 1.0 + 1.0j], [2.0 - 1.0j, -1.0 + 0.0j]],
        dtype=np.complex64,
    )
    input_dict = {
        "operand": operand,
        "dimension": -1,
        "is_stable": False,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array
    operand = np.array([True, False, True, True, False, False], dtype=bool)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D uint8 array, sort along dimension 0
    operand = np.random.randint(0, 256, size=(6, 4), dtype=np.uint8)
    input_dict = {
        "operand": operand,
        "dimension": 0,
        "is_stable": False,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D float32 array, sort along negative dimension
    operand = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "dimension": -2,
        "is_stable": True,
        "num_keys": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.sort_1"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_1'.")


check_valid('jax.lax.sort', generated_inputs['jax.lax.sort_1'], lib="jax", suffix=1)
