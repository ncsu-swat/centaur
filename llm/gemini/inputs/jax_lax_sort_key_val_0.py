
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sort_key_val_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, default dimension (-1)
    keys = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=np.float32)
    values = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": -1,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 arrays, dimension 0, not stable
    keys = np.array([10, -5, 0, 20, -15], dtype=np.int32)
    values = np.array([100, 200, 300, 400, 500], dtype=np.int32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 0,
        "is_stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 arrays, sorting along axis 0
    keys = np.random.randn(4, 3).astype(np.float32)
    values = np.arange(12).reshape(4, 3).astype(np.float32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 0,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 arrays, sorting along axis 1
    keys = np.random.randn(3, 5).astype(np.float64)
    values = np.random.randn(3, 5).astype(np.float64)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 1,
        "is_stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 arrays, sorting along axis 2
    keys = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    values = np.random.randint(0, 1000, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 2,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 arrays, sorting along negative axis (-1)
    keys = np.random.uniform(-10.0, 10.0, size=(2, 2, 5)).astype(np.float32)
    values = np.random.uniform(-10.0, 10.0, size=(2, 2, 5)).astype(np.float32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": -1,
        "is_stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int32 arrays, sorting along axis 1
    keys = np.random.randint(0, 50, size=(2, 3, 2, 2)).astype(np.int32)
    values = np.random.randint(0, 50, size=(2, 3, 2, 2)).astype(np.int32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 1,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 arrays, sorting along axis 0
    keys = np.array([400, 100, 300, 200], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 0,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 arrays, sorting along axis -2
    keys = np.random.randn(3, 3).astype(np.float32)
    values = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": -2,
        "is_stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D float32 array
    keys = np.random.randn(100).astype(np.float32)
    values = np.arange(100).astype(np.float32)
    input_dict = {
        "keys": keys,
        "values": values,
        "dimension": 0,
        "is_stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.sort_key_val"] = sort_key_val_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_key_val' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_key_val'.")


check_valid('jax.lax.sort_key_val', generated_inputs['jax.lax.sort_key_val'], lib="jax", suffix=0)
