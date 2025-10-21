
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_not_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    out1 = torch.tensor(np.zeros_like(input1)).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([0, 1, 0, 1]).numpy()
    out2 = torch.tensor(np.zeros_like(input2)).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([[-1, -2], [-3, -4]]).numpy()
    out3 = torch.tensor(np.zeros_like(input3)).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1, 0, 1], [0, 1, 0]]).numpy()
    out4 = torch.tensor(np.zeros_like(input4)).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([255, 0, 127]).numpy()
    out5 = torch.tensor(np.zeros_like(input5)).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    out6 = torch.tensor(np.zeros_like(input6)).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([[1, 2], [3, 4]]).numpy()
    out7 = torch.tensor(np.zeros_like(input7)).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([0]).numpy()
    out8 = torch.tensor(np.zeros_like(input8)).numpy()
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([100]).numpy()
    out9 = torch.tensor(np.zeros_like(input9)).numpy()
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randint(0, 256, (3, 3)).numpy()
    out10 = torch.tensor(np.zeros_like(input10)).numpy()
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_not'.")


check_valid('torch.bitwise_not', generated_inputs['torch.bitwise_not'], lib="torch", suffix=0)
