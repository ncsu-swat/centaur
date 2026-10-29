
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def approx_max_k_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default parameters
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

    # Input 2: 2D float32 array, reduction along dimension 1
    operand = np.random.randn(32, 128).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 10,
        "reduction_dimension": 1,
        "recall_target": 0.90,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, reduction along dimension 0, aggregate_to_topk=False
    operand = np.random.randn(64, 16).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 8,
        "reduction_dimension": 0,
        "recall_target": 0.85,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, reduction along last dimension
    operand = np.random.randn(4, 8, 64).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 16,
        "reduction_dimension": -1,
        "recall_target": 0.99,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float16 array, small k
    operand = np.random.randn(50).astype(np.float16)
    input_dict = {
        "operand": operand,
        "k": 1,
        "reduction_dimension": 0,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array with reduction_input_size_override set
    operand = np.random.randn(10, 256).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 20,
        "reduction_dimension": 1,
        "recall_target": 0.80,
        "reduction_input_size_override": 512,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array with negative values only
    operand = -np.random.exponential(scale=2.0, size=(16, 80)).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 5,
        "reduction_dimension": -1,
        "recall_target": 0.92,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, reduction along middle dimension
    operand = np.random.randn(8, 48, 12).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 4,
        "reduction_dimension": 1,
        "recall_target": 0.75,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array
    operand = np.random.randn(2, 4, 8, 32).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 8,
        "reduction_dimension": 3,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, negative dimension indexing (-2)
    operand = np.random.randn(100, 20).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 15,
        "reduction_dimension": -2,
        "recall_target": 0.90,
        "reduction_input_size_override": 200,
        "aggregate_to_topk": True,
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
