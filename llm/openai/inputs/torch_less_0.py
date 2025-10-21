
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_less_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    tensor1_1 = torch.tensor([4, 5, 6]).numpy()
    out1 = torch.empty(0).numpy()
    input_dict1 = {"input": input1, "other": tensor1_1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([-1, -2, -3]).numpy()
    tensor2_1 = torch.tensor([-4, -5, -6]).numpy()
    out2 = torch.empty(0).numpy()
    input_dict2 = {"input": input2, "other": tensor2_1, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor3_1 = torch.tensor([1.1, 2.2, 3.3]).numpy()
    out3 = torch.empty(0).numpy()
    input_dict3 = {"input": input3, "other": tensor3_1, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.ones((2, 2)).numpy()
    tensor4_1 = torch.tensor([[2.0, 2.0], [2.0, 2.0]]).numpy()
    out4 = torch.empty(0).numpy()
    input_dict4 = {"input": input4, "other": tensor4_1, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 4).numpy()
    tensor5_1 = torch.randn(3, 4).numpy()
    out5 = torch.empty(0).numpy()
    input_dict5 = {"input": input5, "other": tensor5_1, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.tensor([0]).numpy()
    tensor6_1 = torch.tensor([1]).numpy()
    out6 = torch.empty(0).numpy()
    input_dict6 = {"input": input6, "other": tensor6_1, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([5, 5, 5]).numpy()
    tensor7_1 = torch.tensor([5, 5, 5]).numpy()
    out7 = torch.empty(0).numpy()
    input_dict7 = {"input": input7, "other": tensor7_1, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.less"] = torch_less_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.less' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less'.")


check_valid('torch.less', generated_inputs['torch.less'], lib="torch", suffix=0)
