
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cosine_similarity_inputs():
    list_of_inputs = []
    
    x1_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2_1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    dim_1 = 0
    eps_1 = 1e-6
    
    input_dict_1 = {
        "x1": x1_1,
        "x2": x2_1,
        "dim": dim_1,
        "eps": eps_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    x1_2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2_2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    dim_2 = 1
    eps_2 = 1e-8
    
    input_dict_2 = {
        "x1": x1_2,
        "x2": x2_2,
        "dim": dim_2,
        "eps": eps_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    x1_3 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x2_3 = np.array([-4.0, 5.0, -6.0], dtype=np.float32)
    dim_3 = 0
    eps_3 = 1e-7
    
    input_dict_3 = {
        "x1": x1_3,
        "x2": x2_3,
        "dim": dim_3,
        "eps": eps_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    x1_4 = np.random.rand(5, 10).astype(np.float32)
    x2_4 = np.random.rand(5, 10).astype(np.float32)
    dim_4 = 1
    eps_4 = 1e-5
    
    input_dict_4 = {
        "x1": x1_4,
        "x2": x2_4,
        "dim": dim_4,
        "eps": eps_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    x1_5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2_5 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    dim_5 = 1
    eps_5 = 1e-9
    
    input_dict_5 = {
        "x1": x1_5,
        "x2": x2_5,
        "dim": dim_5,
        "eps": eps_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.cosine_similarity"] = cosine_similarity_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.cosine_similarity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_similarity'.")


check_valid('torch.nn.functional.cosine_similarity', generated_inputs['torch.nn.functional.cosine_similarity'], lib="torch", suffix=0)
