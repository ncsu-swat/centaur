
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def approx_min_k_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, default recall target, aggregate to top-k
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

    # Input 2: 2D array, reduce along axis 1, no aggregation to top-k (unsorted)
    operand = np.random.rand(50, 100).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 10,
        "reduction_dimension": 1,
        "recall_target": 0.90,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 array, negative values, reduce along axis 0
    operand = (np.random.rand(10, 20) * -100.0).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 3,
        "reduction_dimension": 0,
        "recall_target": 0.99,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float32, reducing along middle dimension
    operand = np.random.randn(5, 10, 20).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 2,
        "reduction_dimension": 1,
        "recall_target": 0.85,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using reduction_input_size_override
    operand = np.random.randn(200).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 10,
        "reduction_dimension": 0,
        "recall_target": 0.95,
        "reduction_input_size_override": 400,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 array, low recall target
    operand = np.random.randn(50, 50).astype(np.float16)
    input_dict = {
        "operand": operand,
        "k": 1,
        "reduction_dimension": -1,
        "recall_target": 0.50,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large k, aggregate_to_topk as False
    operand = np.random.randn(300).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 50,
        "reduction_dimension": 0,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Uniformly distributed values with both positive and negative bounds
    operand = np.random.uniform(-10.0, 10.0, (100, 100)).astype(np.float32)
    input_dict = {
        "operand": operand,
        "k": 8,
        "reduction_dimension": 1,
        "recall_target": 0.90,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional array (4D), float64
    operand = np.random.normal(size=(4, 5, 6, 7)).astype(np.float64)
    input_dict = {
        "operand": operand,
        "k": 3,
        "reduction_dimension": 2,
        "recall_target": 0.95,
        "reduction_input_size_override": -1,
        "aggregate_to_topk": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Linear arange sequence converted to float
    operand = np.arange(1000, dtype=np.float32)
    input_dict = {
        "operand": operand,
        "k": 12,
        "reduction_dimension": 0,
        "recall_target": 0.99,
        "reduction_input_size_override": 1000,
        "aggregate_to_topk": False
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
