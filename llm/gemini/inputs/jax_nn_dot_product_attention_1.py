
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dot_product_attention_inputs():
    list_of_inputs = []

    # Input 1: Standard Multi-Head Attention (MHA) with float32
    input_dict_1 = {
        'query': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'key': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'value': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'bias': np.random.randn(2, 4, 8, 8).astype(np.float32),
        'mask': np.ones((2, 4, 8, 8), dtype=np.bool_),
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': np.array([8, 8], dtype=np.int32),
        'key_value_seq_lengths': np.array([8, 8], dtype=np.int32),
        'local_window_size': 4,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multi-Query Attention (MQA) with float16 and causal masking
    input_dict_2 = {
        'query': np.random.randn(1, 16, 8, 32).astype(np.float16),
        'key': np.random.randn(1, 16, 1, 32).astype(np.float16),
        'value': np.random.randn(1, 16, 1, 32).astype(np.float16),
        'bias': np.zeros((1, 8, 16, 16), dtype=np.float16),
        'mask': np.random.choice([True, False], size=(1, 8, 16, 16)).astype(np.bool_),
        'scale': 0.176,
        'is_causal': True,
        'query_seq_lengths': np.array([16], dtype=np.int32),
        'key_value_seq_lengths': np.array([16], dtype=np.int32),
        'local_window_size': 8,
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Grouped Query Attention (GQA) with float64
    input_dict_3 = {
        'query': np.random.randn(3, 12, 8, 16).astype(np.float64),
        'key': np.random.randn(3, 12, 2, 16).astype(np.float64),
        'value': np.random.randn(3, 12, 2, 16).astype(np.float64),
        'bias': np.random.randn(3, 8, 12, 12).astype(np.float64),
        'mask': np.ones((3, 8, 12, 12), dtype=np.bool_),
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': np.array([12, 10, 8], dtype=np.int32),
        'key_value_seq_lengths': np.array([12, 11, 9], dtype=np.int32),
        'local_window_size': 6,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Standard MHA with high head-dimension and return_residual
    input_dict_4 = {
        'query': np.random.randn(1, 4, 2, 64).astype(np.float32),
        'key': np.random.randn(1, 4, 2, 64).astype(np.float32),
        'value': np.random.randn(1, 4, 2, 64).astype(np.float32),
        'bias': np.random.randn(1, 2, 4, 4).astype(np.float32),
        'mask': np.ones((1, 2, 4, 4), dtype=np.bool_),
        'scale': 0.125,
        'is_causal': True,
        'query_seq_lengths': np.array([4], dtype=np.int32),
        'key_value_seq_lengths': np.array([4], dtype=np.int32),
        'local_window_size': 2,
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: MHA with negative bias values and smaller window size
    input_dict_5 = {
        'query': np.random.randn(1, 2, 1, 8).astype(np.float32),
        'key': np.random.randn(1, 2, 1, 8).astype(np.float32),
        'value': np.random.randn(1, 2, 1, 8).astype(np.float32),
        'bias': np.full((1, 1, 2, 2), -10.0, dtype=np.float32),
        'mask': np.ones((1, 1, 2, 2), dtype=np.bool_),
        'scale': 0.353,
        'is_causal': False,
        'query_seq_lengths': np.array([2], dtype=np.int32),
        'key_value_seq_lengths': np.array([2], dtype=np.int32),
        'local_window_size': 1,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large batch with sequence length masking
    input_dict_6 = {
        'query': np.random.randn(4, 6, 4, 16).astype(np.float32),
        'key': np.random.randn(4, 6, 4, 16).astype(np.float32),
        'value': np.random.randn(4, 6, 4, 16).astype(np.float32),
        'bias': np.zeros((4, 4, 6, 6), dtype=np.float32),
        'mask': np.random.choice([True, False], size=(4, 4, 6, 6), p=[0.9, 0.1]).astype(np.bool_),
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': np.array([6, 5, 4, 6], dtype=np.int32),
        'key_value_seq_lengths': np.array([6, 6, 5, 5], dtype=np.int32),
        'local_window_size': 3,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large sequence size with GQA
    input_dict_7 = {
        'query': np.random.randn(2, 32, 4, 32).astype(np.float32),
        'key': np.random.randn(2, 32, 2, 32).astype(np.float32),
        'value': np.random.randn(2, 32, 2, 32).astype(np.float32),
        'bias': np.zeros((2, 4, 32, 32), dtype=np.float32),
        'mask': np.ones((2, 4, 32, 32), dtype=np.bool_),
        'scale': 0.176,
        'is_causal': True,
        'query_seq_lengths': np.array([32, 24], dtype=np.int32),
        'key_value_seq_lengths': np.array([32, 28], dtype=np.int32),
        'local_window_size': 16,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single head large sequence MQA
    input_dict_8 = {
        'query': np.random.randn(1, 64, 1, 16).astype(np.float32),
        'key': np.random.randn(1, 64, 1, 16).astype(np.float32),
        'value': np.random.randn(1, 64, 1, 16).astype(np.float32),
        'bias': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'mask': np.ones((1, 1, 64, 64), dtype=np.bool_),
        'scale': 0.25,
        'is_causal': False,
        'query_seq_lengths': np.array([64], dtype=np.int32),
        'key_value_seq_lengths': np.array([64], dtype=np.int32),
        'local_window_size': 32,
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Standard MHA with float16 and non-trivial local_window_size
    input_dict_9 = {
        'query': np.random.randn(2, 10, 6, 16).astype(np.float16),
        'key': np.random.randn(2, 10, 6, 16).astype(np.float16),
        'value': np.random.randn(2, 10, 6, 16).astype(np.float16),
        'bias': np.random.randn(2, 6, 10, 10).astype(np.float16),
        'mask': np.ones((2, 6, 10, 10), dtype=np.bool_),
        'scale': 0.25,
        'is_causal': True,
        'query_seq_lengths': np.array([10, 9], dtype=np.int32),
        'key_value_seq_lengths': np.array([10, 10], dtype=np.int32),
        'local_window_size': 5,
        'implementation': 'xla',
        'return_residual': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Standard MHA with float32, active mask and custom scale
    input_dict_10 = {
        'query': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'key': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'value': np.random.randn(2, 8, 4, 16).astype(np.float32),
        'bias': np.random.uniform(-1.0, 1.0, (2, 4, 8, 8)).astype(np.float32),
        'mask': np.random.choice([True, False], size=(2, 4, 8, 8)).astype(np.bool_),
        'scale': 0.3,
        'is_causal': False,
        'query_seq_lengths': np.array([8, 6], dtype=np.int32),
        'key_value_seq_lengths': np.array([8, 7], dtype=np.int32),
        'local_window_size': 3,
        'implementation': 'xla',
        'return_residual': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
