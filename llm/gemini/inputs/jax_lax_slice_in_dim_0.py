
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard slice
    operand = np.arange(10, dtype=np.float32)
    input_dict = {
        "operand": operand,
        "start_index": 2,
        "limit_index": 7,
        "stride": 1,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, stride > 1 along axis 0
    operand = np.arange(20, dtype=np.int32).reshape(5, 4)
    input_dict = {
        "operand": operand,
        "start_index": 0,
        "limit_index": 5,
        "stride": 2,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, slice along axis 1
    operand = np.random.randn(4, 6).astype(np.float64)
    input_dict = {
        "operand": operand,
        "start_index": 1,
        "limit_index": 4,
        "stride": 1,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, negative axis
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_index": 0,
        "limit_index": 3,
        "stride": 1,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array, strided slice along negative axis
    operand = np.random.randint(0, 100, size=(6, 8, 4), dtype=np.int32)
    input_dict = {
        "operand": operand,
        "start_index": 1,
        "limit_index": 7,
        "stride": 2,
        "axis": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array, slice along axis 2
    operand = np.random.choice([True, False], size=(2, 3, 6, 2))
    input_dict = {
        "operand": operand,
        "start_index": 0,
        "limit_index": 5,
        "stride": 1,
        "axis": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64 array, large stride
    operand = np.arange(100, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "start_index": 10,
        "limit_index": 90,
        "stride": 15,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array, slice first dimension
    operand = np.random.randn(4, 2, 2, 3, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "start_index": 1,
        "limit_index": 3,
        "stride": 1,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, zero-length slice (start == limit)
    operand = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "operand": operand,
        "start_index": 2,
        "limit_index": 2,
        "stride": 1,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int32 array, axis 1 slice with large stride
    operand = np.arange(60, dtype=np.int32).reshape(2, 10, 3)
    input_dict = {
        "operand": operand,
        "start_index": 0,
        "limit_index": 10,
        "stride": 3,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D int64 array, slicing last axis with negative indexing
    operand = np.ones((2, 3, 4, 8), dtype=np.int64)
    input_dict = {
        "operand": operand,
        "start_index": 2,
        "limit_index": 6,
        "stride": 2,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.slice_in_dim"] = slice_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_in_dim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_in_dim'.")


check_valid('jax.lax.slice_in_dim', generated_inputs['jax.lax.slice_in_dim'], lib="jax", suffix=0)
