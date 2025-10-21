
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def parse_type_comment_inputs():
    list_of_inputs = []
    
    comment1 = "torch.Tensor"
    list_of_inputs.append({"comment": comment1})
    
    comment2 = "torch.float32"
    list_of_inputs.append({"comment": comment2})
    
    comment3 = "torch.int64"
    list_of_inputs.append({"comment": comment3})
    
    return list_of_inputs

generated_inputs["torch.parse_type_comment"] = parse_type_comment_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.parse_type_comment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_type_comment'.")


check_valid('torch.parse_type_comment', generated_inputs['torch.parse_type_comment'], lib="torch", suffix=0)
