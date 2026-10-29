
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dot_product_attention_inputs():
    list_of_inputs = []

    # Signature:
    # {'query': 'tensor', 'key': 'tensor', 'value': 'tensor', 'bias': 'tensor', 'mask': 'tensor',
    #  'scale': 'float', 'is_causal': 'boolean', 'query_seq_lengths': 'tensor',
    #  'key_value_seq_lengths': 'tensor', 'local_window_size': 'integer', 'implementation': 'string',
    #  'return_residual': 'boolean'}

    # Case 1: Standard 4D input (B, T, N, H) with B=2, T=4, S=4, N=2, K=2, H=8
    B, T, S, N, K, H = 2, 4, 4, 2, 2, 8
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((B, N, T, S), dtype=np.float32),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": float(1.0 / np.sqrt(H)),
        "is_causal": False,
        "query_seq_lengths": np.array([4, 4], dtype=np.int32),
        "key_value_seq_lengths": np.array([4, 4], dtype=np.int32),
        "local_window_size": 2,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 2: 4D input with causal=True, return_residual=True
    B, T, S, N, K, H = 1, 8, 8, 4, 4, 16
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.random.randn(B, N, T, S).astype(np.float32),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": 0.25,
        "is_causal": True,
        "query_seq_lengths": np.array([8], dtype=np.int32),
        "key_value_seq_lengths": np.array([8], dtype=np.int32),
        "local_window_size": 4,
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 3: Grouped Query Attention (GQA), N=4, K=2 (G=2)
    B, T, S, N, K, H = 2, 6, 6, 4, 2, 16
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((B, N, T, S), dtype=np.float32),
        "mask": np.ones((B, 1, T, S), dtype=bool),
        "scale": 0.5,
        "is_causal": False,
        "query_seq_lengths": np.array([6, 6], dtype=np.int32),
        "key_value_seq_lengths": np.array([6, 6], dtype=np.int32),
        "local_window_size": 3,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 4: Multi-Query Attention (MQA), N=4, K=1
    B, T, S, N, K, H = 3, 5, 5, 4, 1, 8
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((1, 1, T, S), dtype=np.float32),
        "mask": np.ones((1, 1, T, S), dtype=bool),
        "scale": float(1.0 / np.sqrt(H)),
        "is_causal": False,
        "query_seq_lengths": np.array([5, 5, 5], dtype=np.int32),
        "key_value_seq_lengths": np.array([5, 5, 5], dtype=np.int32),
        "local_window_size": 1,
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 5: Float64 dtype
    B, T, S, N, K, H = 1, 4, 4, 2, 2, 8
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float64),
        "key": np.random.randn(B, S, K, H).astype(np.float64),
        "value": np.random.randn(B, S, K, H).astype(np.float64),
        "bias": np.random.randn(B, N, T, S).astype(np.float64),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": 0.125,
        "is_causal": False,
        "query_seq_lengths": np.array([4], dtype=np.int32),
        "key_value_seq_lengths": np.array([4], dtype=np.int32),
        "local_window_size": 2,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 6: Unequal query and key/value sequence lengths (T != S)
    B, T, S, N, K, H = 2, 3, 7, 2, 2, 16
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((B, N, T, S), dtype=np.float32),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": 0.25,
        "is_causal": False,
        "query_seq_lengths": np.array([3, 2], dtype=np.int32),
        "key_value_seq_lengths": np.array([7, 5], dtype=np.int32),
        "local_window_size": 5,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 7: Non-trivial boolean mask with masked elements
    B, T, S, N, K, H = 1, 4, 4, 1, 1, 8
    mask = np.array([[[[True, True, False, False],
                       [True, True, True, False],
                       [True, True, True, True],
                       [True, True, True, True]]]], dtype=bool)
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((B, N, T, S), dtype=np.float32),
        "mask": mask,
        "scale": 0.35,
        "is_causal": False,
        "query_seq_lengths": np.array([4], dtype=np.int32),
        "key_value_seq_lengths": np.array([4], dtype=np.int32),
        "local_window_size": 2,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 8: Broadcastable bias of shape (1, 1, T, S)
    B, T, S, N, K, H = 2, 4, 4, 4, 4, 8
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.random.randn(1, 1, T, S).astype(np.float32),
        "mask": np.ones((1, 1, T, S), dtype=bool),
        "scale": 0.5,
        "is_causal": True,
        "query_seq_lengths": np.array([4, 3], dtype=np.int32),
        "key_value_seq_lengths": np.array([4, 3], dtype=np.int32),
        "local_window_size": 3,
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 9: Larger dimensions and window size
    B, T, S, N, K, H = 2, 16, 16, 8, 4, 32
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": np.zeros((B, N, T, S), dtype=np.float32),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": float(1.0 / np.sqrt(H)),
        "is_causal": False,
        "query_seq_lengths": np.array([16, 16], dtype=np.int32),
        "key_value_seq_lengths": np.array([16, 16], dtype=np.int32),
        "local_window_size": 8,
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    # Case 10: Negative bias values and smaller window size
    B, T, S, N, K, H = 2, 5, 5, 2, 2, 4
    inp = {
        "query": np.random.randn(B, T, N, H).astype(np.float32),
        "key": np.random.randn(B, S, K, H).astype(np.float32),
        "value": np.random.randn(B, S, K, H).astype(np.float32),
        "bias": -100.0 * np.random.rand(B, N, T, S).astype(np.float32),
        "mask": np.ones((B, N, T, S), dtype=bool),
        "scale": 1.0,
        "is_causal": False,
        "query_seq_lengths": np.array([5, 5], dtype=np.int32),
        "key_value_seq_lengths": np.array([5, 5], dtype=np.int32),
        "local_window_size": 1,
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(inp))

    return list_of_inputs


generated_inputs["jax.nn.dot_product_attention_1"] = dot_product_attention_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.dot_product_attention_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.dot_product_attention_1'.")


check_valid('jax.nn.dot_product_attention', generated_inputs['jax.nn.dot_product_attention_1'], lib="jax", suffix=1)
