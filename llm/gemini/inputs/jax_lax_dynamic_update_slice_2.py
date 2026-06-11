
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    operand = np.zeros(10, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = (2,)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32
    operand = np.zeros((5, 5), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = (1, 1)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32, allow_negative_indices=False
    operand = np.arange(64, dtype=np.int32).reshape((4, 4, 4))
    update = np.zeros((2, 2, 2), dtype=np.int32)
    start_indices = (1, 2, 0)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative indices with allow_negative_indices=True
    operand = np.random.randn(8, 8).astype(np.float64)
    update = np.random.randn(3, 3).astype(np.float64)
    start_indices = (-4, -4)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Out of bounds indices (clamping test)
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = (5,)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D complex64
    operand = np.zeros((3, 3, 3, 3), dtype=np.complex64)
    update = np.ones((1, 1, 1, 1), dtype=np.complex64)
    start_indices = (0, 1, 2, 0)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D bool
    operand = np.zeros(100, dtype=bool)
    update = np.ones(10, dtype=bool)
    start_indices = (50,)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D uint8
    operand = np.random.randint(0, 255, size=(10, 20), dtype=np.uint8)
    update = np.random.randint(0, 255, size=(5, 5), dtype=np.uint8)
    start_indices = (2, 10)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float16
    operand = np.zeros((2, 2, 2, 2, 2), dtype=np.float16)
    update = np.ones((1, 1, 1, 1, 1), dtype=np.float16)
    start_indices = (0, 0, 0, 0, 0)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64 with mixed dimension update
    operand = np.zeros((10, 10, 10), dtype=np.int64)
    update = np.ones((1, 5, 2), dtype=np.int64)
    start_indices = (9, 2, 5)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_2"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_2'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_2'], lib="jax", suffix=2)
