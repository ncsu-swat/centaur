
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive start_index, axis 0
    operand = np.zeros((10,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 2,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32, negative start_index
    operand = np.arange(8, dtype=np.int32)
    update = np.array([99, 100], dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": -3,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64, axis 0, update smaller along both dimensions
    operand = np.zeros((6, 8), dtype=np.float64)
    update = np.ones((2, 4), dtype=np.float64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 1,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64, axis 1, update matching dimension along axis 0
    operand = np.ones((4, 6), dtype=np.int64)
    update = np.full((4, 2), 5, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 3,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32, axis 2, start_index at 0
    operand = np.zeros((3, 4, 5), dtype=np.float32)
    update = np.ones((2, 3, 2), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 0,
        "axis": 2,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D bool array, axis 1, negative start_index
    operand = np.zeros((2, 5, 3), dtype=np.bool_)
    update = np.ones((2, 2, 3), dtype=np.bool_)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": -2,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex64 array, axis 0
    operand = np.zeros((5, 3, 4, 2), dtype=np.complex64)
    update = np.ones((2, 2, 2, 2), dtype=np.complex64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 2,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D uint8, negative axis specification
    operand = np.zeros((4, 7), dtype=np.uint8)
    update = np.full((3, 2), 255, dtype=np.uint8)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 4,
        "axis": -1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32, start_index exceeding bounds (lax clamps)
    operand = np.zeros((5,), dtype=np.float32)
    update = np.ones((2,), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 10,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int16, negative axis and negative start_index
    operand = np.zeros((4, 6, 8), dtype=np.int16)
    update = np.ones((2, 3, 4), dtype=np.int16)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": -4,
        "axis": -2,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D float64, axis 3
    operand = np.zeros((2, 3, 4, 6), dtype=np.float64)
    update = np.ones((2, 2, 3, 2), dtype=np.float64)
    input_dict = {
        "operand": operand,
        "update": update,
        "start_index": 1,
        "axis": 3,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_in_dim_1"] = (
    dynamic_update_slice_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_in_dim_1'.")


check_valid('jax.lax.dynamic_update_slice_in_dim', generated_inputs['jax.lax.dynamic_update_slice_in_dim_1'], lib="jax", suffix=1)
