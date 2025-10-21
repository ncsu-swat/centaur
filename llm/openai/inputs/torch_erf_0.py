
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def erf_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([0.0, 1.0, -1.0]).numpy()
    out1 = torch.empty(input1.shape).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([1.5, -2.5, 3.0]).numpy()
    out2 = torch.empty(input2.shape).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[0.0, 1.0], [-1.0, 2.0]]).numpy()
    out3 = torch.empty(input3.shape).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
    out4 = torch.empty(input4.shape).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 3, 4).numpy()
    out5 = torch.empty(input5.shape).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.tensor([np.pi, np.e, np.sqrt(2)]).numpy()
    out6 = torch.empty(input6.shape).numpy()
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([-np.pi, -np.e, -np.sqrt(2)]).numpy()
    out7 = torch.empty(input7.shape).numpy()
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([0.5, -0.5, 1.5, -1.5]).numpy()
    out8 = torch.empty(input8.shape).numpy()
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    out9 = torch.empty(input9.shape).numpy()
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(3).numpy()
    out10 = torch.empty(input10.shape).numpy()
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.erf"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erf'.")


check_valid('torch.erf', generated_inputs['torch.erf'], lib="torch", suffix=0)
