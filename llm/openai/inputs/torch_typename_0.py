
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def typename_inputs():
    list_of_inputs = []

    input1 = {"obj": torch.tensor([1, 2, 3]).numpy()}
    list_of_inputs.append(input1)

    input2 = {"obj": torch.tensor([[1, 2], [3, 4]]).numpy()}
    list_of_inputs.append(input2)

    input3 = {"obj": torch.tensor([1.0, 2.0, 3.0]).numpy()}
    list_of_inputs.append(input3)

    input4 = {"obj": torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()}
    list_of_inputs.append(input4)

    input5 = {"obj": torch.tensor([-1, -2, -3]).numpy()}
    list_of_inputs.append(input5)

    input6 = {"obj": torch.tensor([[-1, -2], [-3, -4]]).numpy()}
    list_of_inputs.append(input6)

    input7 = {"obj": torch.randn(2, 3, 4).numpy()}
    list_of_inputs.append(input7)

    input8 = {"obj": torch.randint(0, 10, (5,)).numpy()}
    list_of_inputs.append(input8)

    input9 = {"obj": torch.zeros(3, 3).numpy()}
    list_of_inputs.append(input9)

    input10 = {"obj": torch.ones(2, 2, 2).numpy()}
    list_of_inputs.append(input10)

    return list_of_inputs

generated_inputs["torch.typename"] = typename_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.typename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.typename'.")


check_valid('torch.typename', generated_inputs['torch.typename'], lib="torch", suffix=0)
