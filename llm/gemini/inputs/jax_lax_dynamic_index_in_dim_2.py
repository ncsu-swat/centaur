
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, basic positive index
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {
        "operand": operand,
        "index": 2,
        "axis": 0,
        "keepdims": True,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, keepdims=False
    operand = np.arange(10, dtype=np.int32)
    input_dict = {
        "operand": operand,
        "index": 0,
        "axis": 0,
        "keepdims": False,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, indexing along axis 0 with negative index
    operand = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        "operand": operand,
        "index": -1,
        "axis": 0,
        "keepdims": True,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, indexing along axis 1 with keepdims=False
    operand = np.arange(20, dtype=np.int64).reshape(4, 5)
    input_dict = {
        "operand": operand,
        "index": 3,
        "axis": 1,
        "keepdims": False,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, indexing along axis 2
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "index": 1,
        "axis": 2,
        "keepdims": True,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex64 array, indexing along axis 1 with negative index
    operand = (
        np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)
    ).astype(np.complex64)
    input_dict = {
        "operand": operand,
        "index": -2,
        "axis": 1,
        "keepdims": False,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D uint8 array, indexing along axis 0
    operand = np.random.randint(0, 255, size=(2, 2, 3, 3), dtype=np.uint8)
    input_dict = {
        "operand": operand,
        "index": 1,
        "axis": 0,
        "keepdims": False,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, indexing along axis 3 with negative index
    operand = np.random.randn(2, 4, 6, 8).astype(np.float32)
    input_dict = {
        "operand": operand,
        "index": -4,
        "axis": 3,
        "keepdims": True,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array, indexing along axis 0
    operand = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    input_dict = {
        "operand": operand,
        "index": 0,
        "axis": 0,
        "keepdims": False,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, indexing along axis 2
    operand = np.random.randn(2, 3, 4, 2, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "index": 2,
        "axis": 2,
        "keepdims": True,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_index_in_dim_2"] = (
    dynamic_index_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_index_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_index_in_dim_2'.")


check_valid('jax.lax.dynamic_index_in_dim', generated_inputs['jax.lax.dynamic_index_in_dim_2'], lib="jax", suffix=2)
