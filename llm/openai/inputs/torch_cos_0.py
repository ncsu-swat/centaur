
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_cos_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out1 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out2 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3).numpy()
    out3 = torch.tensor(np.zeros((2, 3))).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([0.0]).numpy()
    out4 = torch.tensor([0.0]).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([np.pi / 2]).numpy()
    out5 = torch.tensor([0.0]).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(5).numpy()
    out6 = torch.tensor(np.zeros(5)).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1.0, -1.0, 2.0, -2.0]).numpy()
    out7 = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 2, 2).numpy()
    out8 = torch.tensor(np.zeros((2, 2, 2))).numpy()
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out9 = torch.tensor([1.0, 1.0, 1.0]).numpy()
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([2 * np.pi]).numpy()
    out10 = torch.tensor([1.0]).numpy()
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.cos"] = torch_cos_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cos'.")


check_valid('torch.cos', generated_inputs['torch.cos'], lib="torch", suffix=0)
