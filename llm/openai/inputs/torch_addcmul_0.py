
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addcmul_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3)
    tensor1_1 = torch.randn(2, 3)
    tensor2_1 = torch.randn(2, 3)
    value1 = 0.5
    out1 = torch.empty_like(input1).numpy()
    input_dict1 = {"input": input1.numpy(), "tensor1": tensor1_1.numpy(), "tensor2": tensor2_1.numpy(), "value": value1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 2, 2)
    tensor1_2 = torch.randn(3, 2, 2)
    tensor2_2 = torch.randn(3, 2, 2)
    value2 = -0.2
    out2 = torch.empty_like(input2).numpy()
    input_dict2 = {"input": input2.numpy(), "tensor1": tensor1_2.numpy(), "tensor2": tensor2_2.numpy(), "value": value2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 4)
    tensor1_3 = torch.randn(1, 4)
    tensor2_3 = torch.randn(1, 4)
    value3 = 1.0
    out3 = torch.empty_like(input3).numpy()
    input_dict3 = {"input": input3.numpy(), "tensor1": tensor1_3.numpy(), "tensor2": tensor2_3.numpy(), "value": value3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2)
    tensor1_4 = torch.randn(2, 2)
    tensor2_4 = torch.randn(2, 2)
    value4 = -1.5
    out4 = torch.empty_like(input4).numpy()
    input_dict4 = {"input": input4.numpy(), "tensor1": tensor1_4.numpy(), "tensor2": tensor2_4.numpy(), "value": value4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.addcmul"] = addcmul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addcmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addcmul'.")


check_valid('torch.addcmul', generated_inputs['torch.addcmul'], lib="torch", suffix=0)
