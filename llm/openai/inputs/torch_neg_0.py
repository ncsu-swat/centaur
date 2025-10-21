
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_neg_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy().astype(np.float32)
    out1 = torch.empty(input1.shape, dtype=torch.float32)
    input_dict1 = {"input": input1, "out": out1.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.0, -2.0, -3.0]).numpy().astype(np.float32)
    out2 = torch.empty(input2.shape, dtype=torch.float32)
    input_dict2 = {"input": input2, "out": out2.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5).numpy().astype(np.float32)
    out3 = torch.empty(input3.shape, dtype=torch.float32)
    input_dict3 = {"input": input3, "out": out3.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1, 2], [3, 4]]).numpy().astype(np.float32)
    out4 = torch.empty(input4.shape, dtype=torch.float32)
    input_dict4 = {"input": input4, "out": out4.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy().astype(np.float32)
    out5 = torch.empty(input5.shape, dtype=torch.float32)
    input_dict5 = {"input": input5, "out": out5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.zeros(2, 3).numpy().astype(np.float32)
    out6 = torch.empty(input6.shape, dtype=torch.float32)
    input_dict6 = {"input": input6, "out": out6.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.ones(3, 2).numpy().astype(np.float32)
    out7 = torch.empty(input7.shape, dtype=torch.float32)
    input_dict7 = {"input": input7, "out": out7.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.arange(10).numpy().astype(np.float32)
    out8 = torch.empty(input8.shape, dtype=torch.float32)
    input_dict8 = {"input": input8, "out": out8.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 2, 2).numpy().astype(np.float32)
    out9 = torch.empty(input9.shape, dtype=torch.float32)
    input_dict9 = {"input": input9, "out": out9.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([0.0]).numpy().astype(np.float32)
    out10 = torch.empty(input10.shape, dtype=torch.float32)
    input_dict10 = {"input": input10, "out": out10.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.neg"] = torch_neg_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.neg'.")


check_valid('torch.neg', generated_inputs['torch.neg'], lib="torch", suffix=0)
