
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    operand = np.arange(10, dtype=np.float32)
    update = np.array([99.0, 99.0], dtype=np.float32)
    start_indices = [3]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32, negative start index
    operand = np.zeros(5, dtype=np.int32)
    update = np.ones(2, dtype=np.int32)
    start_indices = [-2]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, standard update
    operand = np.random.randn(4, 4).astype(np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = [1, 1]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64
    operand = np.random.randn(5, 6).astype(np.float64)
    update = np.zeros((1, 3), dtype=np.float64)
    start_indices = [2, 3]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32
    operand = np.random.randn(3, 3, 3).astype(np.float32)
    update = np.ones((1, 2, 1), dtype=np.float32)
    start_indices = [0, 1, 2]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int64
    operand = np.zeros((2, 2, 2, 2), dtype=np.int64)
    update = np.ones((1, 1, 1, 1), dtype=np.int64)
    start_indices = [1, 0, 1, 0]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D boolean
    operand = np.zeros(8, dtype=bool)
    update = np.ones(4, dtype=bool)
    start_indices = [4]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 with index that will trigger clamping/fit adjustment
    operand = np.random.randn(3, 3).astype(np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = [2, 2]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 with multiple negative indices
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    update = np.ones((2, 2, 2), dtype=np.float32)
    start_indices = [-3, -2, -1]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    update = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    start_indices = [1, 1, 0, 0, 1]
    input_dict = {
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_1"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_1'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_1'], lib="jax", suffix=1)
