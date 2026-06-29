
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def approx_max_k_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    operand = np.random.randn(100).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 5,
        "reduction_dimension": -1,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, reduction along dimension 0, aggregate_to_topk=False
    operand = np.random.randn(50, 10).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 10,
        "reduction_dimension": 0,
        "recall_target": 0.9,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, reduction along dimension 1
    operand = np.random.randn(5, 20, 8).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 2,
        "reduction_dimension": 1,
        "recall_target": 0.85,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, size override enabled
    operand = np.random.randn(10, 100).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 8,
        "reduction_dimension": -1,
        "recall_target": 0.99,
        "reduction_input_size_override": 1000,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D array, low recall target, aggregate_to_topk=False
    operand = np.random.randn(1000).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 50,
        "reduction_dimension": 0,
        "recall_target": 0.5,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array
    operand = np.random.randn(2, 3, 30, 4).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 3,
        "reduction_dimension": 2,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with all negative values
    operand = -np.abs(np.random.randn(20, 30)).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 5,
        "reduction_dimension": 1,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, reduction_dimension=-2
    operand = np.random.randn(4, 15, 6).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 1,
        "reduction_dimension": -2,
        "recall_target": 0.99,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, size override is positive
    operand = np.random.randn(50).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 1,
        "reduction_dimension": 0,
        "recall_target": 0.95,
        "reduction_input_size_override": 100,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, small sizes
    operand = np.random.randn(2, 2, 2, 10, 2).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 4,
        "reduction_dimension": 3,
        "recall_target": 0.9,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.approx_max_k"] = approx_max_k_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.approx_max_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.approx_max_k'.")


check_valid('jax.lax.approx_max_k', generated_inputs['jax.lax.approx_max_k'], lib="jax", suffix=0)
