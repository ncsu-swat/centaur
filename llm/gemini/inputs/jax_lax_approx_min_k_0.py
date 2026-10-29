
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def approx_min_k_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default parameters style
    operand = np.random.randn(100).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 5,
        "reduction_dimension": -1,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, reduction on dimension 1
    operand = np.random.randn(32, 256).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 10,
        "reduction_dimension": 1,
        "recall_target": 0.9,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, reduction on dimension 0
    operand = np.random.randn(128, 64).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 8,
        "reduction_dimension": 0,
        "recall_target": 0.8,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, reduction_dimension = -1, float16
    operand = np.random.randn(4, 16, 128).astype(np.float16)
    input_dict = {
        "operand": operand,
        "k": 4,
        "reduction_dimension": -1,
        "recall_target": 0.99,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, middle reduction dimension (1)
    operand = np.random.randn(8, 64, 16).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 16,
        "reduction_dimension": 1,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 type array, aggregate_to_topk=False
    operand = np.random.randn(20, 200).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 1,
        "reduction_dimension": -1,
        "recall_target": 0.5,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: reduction_input_size_override specified
    operand = np.random.randn(16, 128).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 5,
        "reduction_dimension": 1,
        "recall_target": 0.9,
        "reduction_input_size_override": 256,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array with negative values
    operand = -np.abs(np.random.randn(2, 4, 8, 64)).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 3,
        "reduction_dimension": 3,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large k relative to dimension size
    operand = np.random.randn(10, 50).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 25,
        "reduction_dimension": -1,
        "recall_target": 0.999,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with aggregate_to_topk=False
    operand = np.random.randn(512).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 20,
        "reduction_dimension": 0,
        "recall_target": 0.75,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.approx_min_k"] = approx_min_k_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.approx_min_k' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.approx_min_k'.")


check_valid('jax.lax.approx_min_k', generated_inputs['jax.lax.approx_min_k'], lib="jax", suffix=0)
