
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.lax


def dynamic_update_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, update with float scalar
    input_dict = {
        "operand": np.zeros((6,), dtype=np.float32),
        "update": float(1.0),
        "index": int(2),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, negative index with allow_negative_indices=True
    input_dict = {
        "operand": np.ones((5,), dtype=np.float64),
        "update": float(-3.5),
        "index": int(-1),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, index out of bounds (clipped)
    input_dict = {
        "operand": np.arange(10, dtype=np.float32),
        "update": float(99.0),
        "index": int(15),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, allow_negative_indices=False
    input_dict = {
        "operand": np.zeros((8,), dtype=np.float32),
        "update": float(4.2),
        "index": int(3),
        "axis": int(0),
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with shape (1,), index 0
    input_dict = {
        "operand": np.array([0.0], dtype=np.float32),
        "update": float(10.0),
        "index": int(0),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, negative index at start
    input_dict = {
        "operand": np.zeros((4,), dtype=np.float64),
        "update": float(2.718),
        "index": int(-4),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, update at index 0
    input_dict = {
        "operand": np.full((12,), 5.0, dtype=np.float32),
        "update": float(0.0),
        "index": int(0),
        "axis": int(0),
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, index at end
    input_dict = {
        "operand": np.ones((7,), dtype=np.float32),
        "update": float(3.1415),
        "index": int(6),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, large negative out of bounds index (clipped)
    input_dict = {
        "operand": np.zeros((5,), dtype=np.float32),
        "update": float(-1.0),
        "index": int(-100),
        "axis": int(0),
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, axis specified as -1 (relative to 1D)
    input_dict = {
        "operand": np.zeros((10,), dtype=np.float64),
        "update": float(7.7),
        "index": int(5),
        "axis": int(-1),
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_index_in_dim_3"] = (
    dynamic_update_index_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_index_in_dim_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_index_in_dim_3'.")


check_valid('jax.lax.dynamic_update_index_in_dim', generated_inputs['jax.lax.dynamic_update_index_in_dim_3'], lib="jax", suffix=3)
