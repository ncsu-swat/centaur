
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, update shape (1,), positive index
    operand = np.zeros((6,), dtype=np.float32)
    update = np.array([5.0], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 2,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, scalar-shaped update, negative index
    operand = np.arange(10, dtype=np.int32)
    update = np.array(99, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": -1,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, update has (ndim - 1) dimensions, axis 0
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((4,), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 1,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, update has same ndim with shape (1, 5), axis 0, allow_negative_indices=False
    operand = np.ones((3, 5), dtype=np.float64)
    update = np.zeros((1, 5), dtype=np.float64)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 2,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, update along axis 1
    operand = np.zeros((5, 3), dtype=np.int64)
    update = np.arange(5, dtype=np.int64)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 1,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, update along axis 0 with ndim - 1
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    update = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 0,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, negative index along axis 1 with explicit size 1 slice
    operand = np.zeros((2, 4, 3), dtype=np.float32)
    update = np.ones((2, 1, 3), dtype=np.float32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": -2,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, negative axis representation (axis=-1)
    operand = np.ones((4, 5, 6), dtype=np.int32)
    update = np.zeros((4, 5, 1), dtype=np.int32)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 3,
        "axis": -1,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D bool array, update matching dimensions
    operand = np.zeros((2, 3, 4, 5), dtype=np.bool_)
    update = np.ones((2, 1, 4, 5), dtype=np.bool_)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 2,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D complex64 array, ndim - 1 update along last axis
    operand = (np.random.randn(2, 3, 2, 2) + 1j * np.random.randn(2, 3, 2, 2)).astype(
        np.complex64
    )
    update = (np.random.randn(2, 3, 2) + 1j * np.random.randn(2, 3, 2)).astype(
        np.complex64
    )
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 0,
        "axis": 3,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D uint8 array with out-of-bounds index (clipped according to documentation)
    operand = np.zeros((5, 5), dtype=np.uint8)
    update = np.full((1, 5), 255, dtype=np.uint8)
    input_dict = {
        "operand": operand,
        "update": update,
        "index": 20,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_index_in_dim_1"] = (
    dynamic_update_index_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_1'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_1'], lib="jax", suffix=1)
