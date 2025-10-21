
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_vander_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3, 5]).numpy()
    N1 = None
    increasing1 = False
    input_dict1 = {"x": input1, "N": N1, "increasing": increasing1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3, 5]).numpy()
    N2 = 3
    increasing2 = False
    input_dict2 = {"x": input2, "N": N2, "increasing": increasing2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1, 2, 3, 5]).numpy()
    N3 = 3
    increasing3 = True
    input_dict3 = {"x": input3, "N": N3, "increasing": increasing3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-1, 0, 1, 2]).numpy()
    N4 = None
    increasing4 = False
    input_dict4 = {"x": input4, "N": N4, "increasing": increasing4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    N5 = 2
    increasing5 = True
    input_dict5 = {"x": input5, "N": N5, "increasing": increasing5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([0.5, 1.5, 2.5]).numpy()
    N6 = 4
    increasing6 = False
    input_dict6 = {"x": input6, "N": N6, "increasing": increasing6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 2, 3]).numpy()
    N7 = None
    increasing7 = True
    input_dict7 = {"x": input7, "N": N7, "increasing": increasing7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([-2, -1, 0, 1]).numpy()
    N8 = 2
    increasing8 = False
    input_dict8 = {"x": input8, "N": N8, "increasing": increasing8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([10, 20, 30]).numpy()
    N9 = 3
    increasing9 = True
    input_dict9 = {"x": input9, "N": N9, "increasing": increasing9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([0, 1, 2, 3, 4]).numpy()
    N10 = None
    increasing10 = True
    input_dict10 = {"x": input10, "N": N10, "increasing": increasing10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.vander"] = torch_vander_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vander'.")


check_valid('torch.vander', generated_inputs['torch.vander'], lib="torch", suffix=0)
