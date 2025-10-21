
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def combinations_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    r1 = 2
    with_replacement1 = False
    input_dict1 = {"input": input1, "r": r1, "with_replacement": with_replacement1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    r2 = 3
    with_replacement2 = True
    input_dict2 = {"input": input2, "r": r2, "with_replacement": with_replacement2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([10, 20, 30]).numpy()
    r3 = 1
    with_replacement3 = False
    input_dict3 = {"input": input3, "r": r3, "with_replacement": with_replacement3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1, 2, 3, 4]).numpy()
    r4 = 4
    with_replacement4 = False
    input_dict4 = {"input": input4, "r": r4, "with_replacement": with_replacement4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([5, 6, 7, 8, 9]).numpy()
    r5 = 2
    with_replacement5 = True
    input_dict5 = {"input": input5, "r": r5, "with_replacement": with_replacement5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.tensor([1, 1, 2, 2, 3, 3]).numpy()
    r6 = 2
    with_replacement6 = False
    input_dict6 = {"input": input6, "r": r6, "with_replacement": with_replacement6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3]).numpy()
    r7 = 0
    with_replacement7 = False
    input_dict7 = {"input": input7, "r": r7, "with_replacement": with_replacement7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([1]).numpy()
    r8 = 1
    with_replacement8 = True
    input_dict8 = {"input": input8, "r": r8, "with_replacement": with_replacement8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    r9 = 5
    with_replacement9 = False
    input_dict9 = {"input": input9, "r": r9, "with_replacement": with_replacement9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([1, 2, 3]).numpy()
    r10 = 2
    with_replacement10 = True
    input_dict10 = {"input": input10, "r": r10, "with_replacement": with_replacement10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.combinations"] = combinations_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.combinations' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.combinations'.")


check_valid('torch.combinations', generated_inputs['torch.combinations'], lib="torch", suffix=0)
