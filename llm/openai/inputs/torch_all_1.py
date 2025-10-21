
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_all_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([True, True, True]).numpy()
    generated_inputs["torch.all_1"] = {"input": input1}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input2 = torch.tensor([False, False, False]).numpy()
    generated_inputs["torch.all_1"] = {"input": input2}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))
    
    input3 = torch.tensor([True, False, True]).numpy()
    generated_inputs["torch.all_1"] = {"input": input3}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input4 = torch.tensor([[True, True], [False, False]]).numpy()
    generated_inputs["torch.all_1"] = {"input": input4}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input5 = torch.tensor([[True, False], [True, True]]).numpy()
    generated_inputs["torch.all_1"] = {"input": input5}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))
    
    input6 = torch.tensor([1, 0, 1]).numpy()
    generated_inputs["torch.all_1"] = {"input": input6}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input7 = torch.tensor([0, 0, 0]).numpy()
    generated_inputs["torch.all_1"] = {"input": input7}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input8 = torch.tensor([-1, -2, -3]).numpy()
    generated_inputs["torch.all_1"] = {"input": input8}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input9 = torch.tensor([1, 1, 1, 1]).numpy()
    generated_inputs["torch.all_1"] = {"input": input9}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))
    
    input10 = torch.tensor([1, 0, 1, 0]).numpy()
    generated_inputs["torch.all_1"] = {"input": input10}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    input11 = torch.randint(0, 2, (3, 3)).numpy()
    generated_inputs["torch.all_1"] = {"input": input11}
    list_of_inputs.append(copy.deepcopy(generated_inputs["torch.all_1"]))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.all_1"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_1'.")


check_valid('torch.all', generated_inputs['torch.all_1'], lib="torch", suffix=1)
