
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def embedding_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)
    weight1 = np.random.rand(10, 3).astype(np.float32)
    padding_idx1 = 0
    max_norm1 = 1.0
    norm_type1 = 2.0
    scale_grad_by_freq1 = False
    sparse1 = False
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "padding_idx": padding_idx1,
        "max_norm": max_norm1,
        "norm_type": norm_type1,
        "scale_grad_by_freq": scale_grad_by_freq1,
        "sparse": sparse1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([0, 2, 0, 5], dtype=np.int64)
    weight2 = np.random.rand(10, 3).astype(np.float32)
    weight2[0, :] = 0.0
    padding_idx2 = 0
    max_norm2 = None
    norm_type2 = 2.0
    scale_grad_by_freq2 = True
    sparse2 = True
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "padding_idx": padding_idx2,
        "max_norm": max_norm2,
        "norm_type": norm_type2,
        "scale_grad_by_freq": scale_grad_by_freq2,
        "sparse": sparse2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    weight3 = np.random.rand(5, 2).astype(np.float32)
    padding_idx3 = 1
    max_norm3 = 0.5
    norm_type3 = 2.0
    scale_grad_by_freq3 = False
    sparse3 = False
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "padding_idx": padding_idx3,
        "max_norm": max_norm3,
        "norm_type": norm_type3,
        "scale_grad_by_freq": scale_grad_by_freq3,
        "sparse": sparse3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 1, 1], dtype=np.int64)
    weight4 = np.random.rand(2, 4).astype(np.float32)
    padding_idx4 = None
    max_norm4 = 2.0
    norm_type4 = 2.0
    scale_grad_by_freq4 = True
    sparse4 = False
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "padding_idx": padding_idx4,
        "max_norm": max_norm4,
        "norm_type": norm_type4,
        "scale_grad_by_freq": scale_grad_by_freq4,
        "sparse": sparse4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding"] = embedding_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.embedding' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding'.")


check_valid('torch.nn.functional.embedding', generated_inputs['torch.nn.functional.embedding'], lib="torch", suffix=0)
