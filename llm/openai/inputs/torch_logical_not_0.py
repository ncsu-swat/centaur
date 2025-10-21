
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_logical_not_inputs():
    list_of_inputs = []

    input1 = torch.tensor([True, False, True, False]).numpy()
    out1 = torch.tensor(np.zeros_like(input1, dtype=bool)).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[True, False], [False, True]]).numpy()
    out2 = torch.tensor(np.zeros_like(input2, dtype=bool)).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([False, False, False]).numpy()
    out3 = torch.tensor(np.zeros_like(input3, dtype=bool)).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([True, True, True]).numpy()
    out4 = torch.tensor(np.zeros_like(input4, dtype=bool)).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([True, False, True, False, True]).numpy()
    out5 = torch.tensor(np.zeros_like(input5, dtype=bool)).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([[True], [False], [True]]).numpy()
    out6 = torch.tensor(np.zeros_like(input6, dtype=bool)).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([False, True, False, True, False, True]).numpy()
    out7 = torch.tensor(np.zeros_like(input7, dtype=bool)).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([True, True, False, False, True, True]).numpy()
    out8 = torch.tensor(np.zeros_like(input8, dtype=bool)).numpy()
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([False, False]).numpy()
    out9 = torch.tensor(np.zeros_like(input9, dtype=bool)).numpy()
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([[True, False, True], [False, True, False]]).numpy()
    out10 = torch.tensor(np.zeros_like(input10, dtype=bool)).numpy()
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.logical_not"] = torch_logical_not_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_not'.")


check_valid('torch.logical_not', generated_inputs['torch.logical_not'], lib="torch", suffix=0)
