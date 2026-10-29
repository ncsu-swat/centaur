
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive index, keepdims=True
    operand = np.arange(10, dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    axis = 0
    keepdims = True
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, positive index along axis 1, keepdims=False
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    index = np.array(1, dtype=np.int64)
    axis = 1
    keepdims = False
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, negative index along axis 0
    operand = np.random.randn(5, 6).astype(np.float64)
    index = np.array(-1, dtype=np.int32)
    axis = 0
    keepdims = True
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axis 2, allow_negative_indices=False
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    index = np.array(0, dtype=np.int32)
    axis = 2
    keepdims = False
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D boolean array, axis 1
    operand = np.random.choice([True, False], size=(2, 4, 3, 5))
    index = np.array(3, dtype=np.int32)
    axis = 1
    keepdims = True
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array, negative index and negative axis
    operand = np.arange(8, dtype=np.int64)
    index = np.array(-3, dtype=np.int32)
    axis = -1
    keepdims = False
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D complex64 array, negative axis=-2
    operand = (
        np.random.randn(4, 5, 2) + 1j * np.random.randn(4, 5, 2)
    ).astype(np.complex64)
    index = np.array(1, dtype=np.int32)
    axis = -2
    keepdims = True
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array, indexing last axis
    operand = np.ones((2, 3, 2, 4, 3), dtype=np.float32)
    index = np.array(0, dtype=np.int64)
    axis = 4
    keepdims = False
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, negative axis with keepdims=True
    operand = np.random.randn(7, 3).astype(np.float32)
    index = np.array(2, dtype=np.int32)
    axis = -1
    keepdims = True
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D uint8 array, negative index along axis 0
    operand = np.random.randint(0, 255, size=(6, 2, 3), dtype=np.uint8)
    index = np.array(-2, dtype=np.int64)
    axis = 0
    keepdims = False
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "index": index,
        "axis": axis,
        "keepdims": keepdims,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_index_in_dim_1"] = (
    dynamic_index_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_index_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_index_in_dim_1'.")


check_valid('jax.lax.dynamic_index_in_dim', generated_inputs['jax.lax.dynamic_index_in_dim_1'], lib="jax", suffix=1)
