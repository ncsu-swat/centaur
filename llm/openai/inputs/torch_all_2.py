
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_all_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([True, True, True]).numpy()
    dim1 = 0
    keepdim1 = False
    out1 = torch.empty(0, dtype=bool).numpy()
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([False, False, False]).numpy()
    dim2 = 0
    keepdim2 = True
    out2 = torch.empty(0, dtype=bool).numpy()
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[True, False], [False, True]]).numpy()
    dim3 = 1
    keepdim3 = False
    out3 = torch.empty(0, dtype=bool).numpy()
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[True, False], [False, True]]).numpy()
    dim4 = 0
    keepdim4 = True
    out4 = torch.empty(0, dtype=bool).numpy()
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1, 0, 1]).numpy()
    dim5 = 0
    keepdim5 = False
    out5 = torch.empty(0, dtype=bool).numpy()
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([[1, 1], [0, 0]]).numpy()
    dim6 = 1
    keepdim6 = False
    out6 = torch.empty(0, dtype=bool).numpy()
    input_dict6 = {"input": input6, "dim": dim6, "keepdim": keepdim6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([[-1, 2], [3, -4]]).numpy()
    dim7 = 0
    keepdim7 = True
    out7 = torch.empty(0, dtype=bool).numpy()
    input_dict7 = {"input": input7, "dim": dim7, "keepdim": keepdim7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 3, 4).numpy()
    dim8 = 2
    keepdim8 = False
    out8 = torch.empty(0, dtype=bool).numpy()
    input_dict8 = {"input": input8, "dim": dim8, "keepdim": keepdim8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randint(0, 2, (5, 5)).numpy()
    dim9 = (0, 1)
    keepdim9 = True
    out9 = torch.empty(0, dtype=bool).numpy()
    input_dict9 = {"input": input9, "dim": dim9, "keepdim": keepdim9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([0, 1, 2, 3]).numpy()
    dim10 = None
    keepdim10 = False
    out10 = torch.empty(0, dtype=bool).numpy()
    input_dict10 = {"input": input10, "dim": dim10, "keepdim": keepdim10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.all_2"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_2'.")


check_valid('torch.all', generated_inputs['torch.all_2'], lib="torch", suffix=2)
