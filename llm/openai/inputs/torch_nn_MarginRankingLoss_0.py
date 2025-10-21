
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(5)
    input2 = torch.randn(5)
    target = torch.randint(-2, 2, (5,)).float()
    
    input_dict = {
        "margin": 0.1,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MarginRankingLoss"] = margin_ranking_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MarginRankingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MarginRankingLoss'.")


check_valid('torch.nn.MarginRankingLoss', generated_inputs['torch.nn.MarginRankingLoss'], lib="torch", suffix=0)
