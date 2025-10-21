
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_lu_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3, 2).numpy()
    input1_dict = {"A": input1, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input1_dict))
    
    input2 = torch.randn(2, 5, 7).numpy()
    input2_dict = {"A": input2, "pivot": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input2_dict))
    
    input3 = torch.randn(2, 2, dtype=torch.float64).numpy()
    input3_dict = {"A": input3, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input3_dict))
    
    input4 = torch.randn(4, 4, dtype=np.float32).numpy()
    input4_dict = {"A": input4, "pivot": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input4_dict))
    
    input5 = torch.randn(1, 3, 3).numpy()
    input5_dict = {"A": input5, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input5_dict))

    input7 = torch.randn(2, 3).numpy()
    input7_dict = {"A": input7, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input7_dict))

    input8 = torch.randn(3, 4).numpy()
    input8_dict = {"A": input8, "pivot": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input8_dict))
    
    input9 = torch.randn(2, 2).numpy()
    input9_dict = {"A": input9, "pivot": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input9_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.lu"] = linalg_lu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.lu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.lu'.")


check_valid('torch.linalg.lu', generated_inputs['torch.linalg.lu'], lib="torch", suffix=0)
