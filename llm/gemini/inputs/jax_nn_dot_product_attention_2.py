
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dot_product_attention_inputs():
    list_of_inputs = []

    # Case 1: Standard Float32 Multi-Head Attention (MHA)
    B, T, S, N, K, H = 2, 8, 8, 4, 4, 16
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([8, 8], dtype=np.int32)
    key_value_seq_lengths = np.array([8, 8], dtype=np.int32)
    
    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (2, 2),
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float16 with Causal Mask
    B, T, S, N, K, H = 1, 4, 4, 2, 2, 8
    query = np.random.randn(B, T, N, H).astype(np.float16)
    key = np.random.randn(B, S, K, H).astype(np.float16)
    value = np.random.randn(B, S, K, H).astype(np.float16)
    bias = np.random.randn(B, N, T, S).astype(np.float16)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([4], dtype=np.int32)
    key_value_seq_lengths = np.array([4], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.35,
        'is_causal': True,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (1, 1),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Grouped Query Attention (GQA) where N is multiple of K
    B, T, S, N, K, H = 2, 6, 6, 4, 2, 16
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([6, 6], dtype=np.int32)
    key_value_seq_lengths = np.array([6, 6], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.125,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (3, 3),
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multi-Query Attention (MQA) where K = 1
    B, T, S, N, K, H = 2, 8, 8, 4, 1, 16
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([8, 8], dtype=np.int32)
    key_value_seq_lengths = np.array([8, 8], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': True,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (2, 2),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger configuration with residual return
    B, T, S, N, K, H = 4, 16, 16, 8, 8, 32
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([16, 16, 16, 16], dtype=np.int32)
    key_value_seq_lengths = np.array([16, 16, 16, 16], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.176,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (4, 4),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Float64 high-precision attention
    B, T, S, N, K, H = 1, 8, 8, 4, 4, 16
    query = np.random.randn(B, T, N, H).astype(np.float64)
    key = np.random.randn(B, S, K, H).astype(np.float64)
    value = np.random.randn(B, S, K, H).astype(np.float64)
    bias = np.random.randn(B, N, T, S).astype(np.float64)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([8], dtype=np.int32)
    key_value_seq_lengths = np.array([8], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (2, 2),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Varied sequence lengths
    B, T, S, N, K, H = 2, 10, 10, 4, 4, 16
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([10, 8], dtype=np.int32)
    key_value_seq_lengths = np.array([10, 9], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': True,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (5, 5),
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Batch of size 3 GQA
    B, T, S, N, K, H = 3, 12, 12, 6, 2, 16
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([12, 12, 12], dtype=np.int32)
    key_value_seq_lengths = np.array([12, 12, 12], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (2, 2),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Small symmetric config without residual
    B, T, S, N, K, H = 2, 4, 4, 2, 2, 8
    query = np.random.randn(B, T, N, H).astype(np.float32)
    key = np.random.randn(B, S, K, H).astype(np.float32)
    value = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([4, 4], dtype=np.int32)
    key_value_seq_lengths = np.array([4, 4], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.353,
        'is_causal': False,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (1, 1),
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Float16 with larger sequence size
    B, T, S, N, K, H = 1, 16, 16, 4, 4, 16
    query = np.random.randn(B, T, N, H).astype(np.float16)
    key = np.random.randn(B, S, K, H).astype(np.float16)
    value = np.random.randn(B, S, K, H).astype(np.float16)
    bias = np.random.randn(B, N, T, S).astype(np.float16)
    mask = np.random.choice([True, False], size=(B, N, T, S))
    query_seq_lengths = np.array([16], dtype=np.int32)
    key_value_seq_lengths = np.array([16], dtype=np.int32)

    input_dict = {
        'query': query,
        'key': key,
        'value': value,
        'bias': bias,
        'mask': mask,
        'scale': 0.25,
        'is_causal': True,
        'query_seq_lengths': query_seq_lengths,
        'key_value_seq_lengths': key_value_seq_lengths,
        'local_window_size': (8, 8),
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.dot_product_attention_2"] = dot_product_attention_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.dot_product_attention_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.dot_product_attention_2'.")


check_valid('jax.nn.dot_product_attention', generated_inputs['jax.nn.dot_product_attention_2'], lib="jax", suffix=2)
