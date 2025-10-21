
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def parse_schema_inputs():
    list_of_inputs = []
    
    schema_string1 = '{"name": "example", "type": "string"}'
    list_of_inputs.append({"schema_string": schema_string1})
    
    return list_of_inputs

generated_inputs["torch.parse_schema"] = parse_schema_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.parse_schema' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_schema'.")


check_valid('torch.parse_schema', generated_inputs['torch.parse_schema'], lib="torch", suffix=0)
