
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def embedding_bag_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64)
    offsets1 = np.array([0, 4], dtype=np.int64)
    per_sample_weights1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    input_dict1 = {
        "num_embeddings": 10,
        "embedding_dim": 3,
        "max_norm": 2.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "include_last_offset": False,
        "padding_idx": None,
        "dtype": torch.long,
        "input": input1,
        "offsets": offsets1,
        "per_sample_weights": per_sample_weights1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.EmbeddingBag"] = embedding_bag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.EmbeddingBag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.EmbeddingBag'.")


check_valid('torch.nn.EmbeddingBag', generated_inputs['torch.nn.EmbeddingBag'], lib="torch", suffix=0)
