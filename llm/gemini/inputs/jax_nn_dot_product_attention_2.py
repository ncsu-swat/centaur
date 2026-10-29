
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dot_product_attention_inputs():
    list_of_inputs = []

    # Input 1: Standard MHA, square sequence length
    B, T, S, N, K, H = 2, 4, 4, 2, 2, 8
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([T, T], dtype=np.int32)
    kv_lens = np.array([S, S], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": float(1.0 / np.sqrt(H)),
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (2, 2),
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grouped Query Attention (GQA), causal, return_residual=True
    B, T, S, N, K, H = 2, 6, 6, 4, 2, 16
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.zeros((B, 1, T, S), dtype=np.float32)
    mask = np.ones((B, 1, T, S), dtype=bool)
    q_lens = np.array([6, 5], dtype=np.int32)
    kv_lens = np.array([6, 5], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.25,
        "is_causal": True,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (6, 6),
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-Query Attention (MQA, K=1)
    B, T, S, N, K, H = 1, 8, 8, 4, 1, 8
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(1, N, T, S).astype(np.float32)
    mask = np.ones((1, N, T, S), dtype=bool)
    q_lens = np.array([8], dtype=np.int32)
    kv_lens = np.array([8], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.125,
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (4, 4),
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-square dimensions T != S
    B, T, S, N, K, H = 2, 3, 5, 2, 2, 4
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.uniform(-1.0, 1.0, size=(B, N, T, S)).astype(np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([3, 2], dtype=np.int32)
    kv_lens = np.array([5, 4], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.5,
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (5, 5),
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Asymmetric local window size
    B, T, S, N, K, H = 3, 4, 4, 2, 2, 8
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.zeros((B, N, T, S), dtype=np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([4, 4, 3], dtype=np.int32)
    kv_lens = np.array([4, 4, 3], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.35,
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (1, 2),
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 precision
    B, T, S, N, K, H = 1, 2, 2, 1, 1, 4
    q = np.random.randn(B, T, N, H).astype(np.float64)
    k = np.random.randn(B, S, K, H).astype(np.float64)
    v = np.random.randn(B, S, K, H).astype(np.float64)
    bias = np.ones((B, N, T, S), dtype=np.float64) * -0.5
    mask = np.array([[[[True, False], [True, True]]]], dtype=bool)
    q_lens = np.array([2], dtype=np.int32)
    kv_lens = np.array([2], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.5,
        "is_causal": True,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (2, 0),
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcastable bias and mask
    B, T, S, N, K, H = 2, 4, 4, 4, 4, 8
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(1, 1, T, S).astype(np.float32)
    mask = np.ones((B, 1, 1, S), dtype=bool)
    q_lens = np.array([4, 4], dtype=np.int32)
    kv_lens = np.array([4, 4], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.2,
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (3, 3),
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger head dimension H=32
    B, T, S, N, K, H = 2, 4, 4, 4, 4, 32
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.uniform(-5.0, 0.0, size=(B, N, T, S)).astype(np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([4, 3], dtype=np.int32)
    kv_lens = np.array([4, 4], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": float(1.0 / np.sqrt(32)),
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (3, 1),
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Grouped Query Attention with K=1 and T=5, S=5
    B, T, S, N, K, H = 2, 5, 5, 2, 1, 16
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.zeros((B, N, T, S), dtype=np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([5, 4], dtype=np.int32)
    kv_lens = np.array([5, 5], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.25,
        "is_causal": True,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (4, 0),
        "implementation": "xla",
        "return_residual": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single batch with smaller scale
    B, T, S, N, K, H = 1, 3, 3, 2, 2, 8
    q = np.random.randn(B, T, N, H).astype(np.float32)
    k = np.random.randn(B, S, K, H).astype(np.float32)
    v = np.random.randn(B, S, K, H).astype(np.float32)
    bias = np.random.randn(B, N, T, S).astype(np.float32)
    mask = np.ones((B, N, T, S), dtype=bool)
    q_lens = np.array([3], dtype=np.int32)
    kv_lens = np.array([3], dtype=np.int32)
    input_dict = {
        "query": q,
        "key": k,
        "value": v,
        "bias": bias,
        "mask": mask,
        "scale": 0.05,
        "is_causal": False,
        "query_seq_lengths": q_lens,
        "key_value_seq_lengths": kv_lens,
        "local_window_size": (2, 2),
        "implementation": "xla",
        "return_residual": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.nn.dot_product_attention_2"] = (
    dot_product_attention_inputs()
)

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
