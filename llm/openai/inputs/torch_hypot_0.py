
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hypot_inputs():
    list_of_inputs = []

    input1 = torch.tensor([3.0]).numpy()
    other1 = torch.tensor([4.0]).numpy()
    out1 = np.array([])
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other2 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    out2 = np.array([])
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    out3 = np.array([])
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-3.0, 4.0]).numpy()
    other4 = torch.tensor([5.0, -12.0]).numpy()
    out4 = np.array([])
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other5 = torch.tensor([4.5, 5.5, 6.5]).numpy()
    out5 = np.array([])
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1.0]).numpy()
    other6 = torch.tensor([2.0, 3.0, 4.0]).numpy()
    out6 = np.array([])
    input_dict6 = {"input": input6, "other": other6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([5.0, 6.0]).numpy()
    other7 = torch.tensor([7.0]).numpy()
    out7 = np.array([])
    input_dict7 = {"input": input7, "other": other7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.hypot"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hypot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hypot'.")


check_valid('torch.hypot', generated_inputs['torch.hypot'], lib="torch", suffix=0)
