
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def floor__inputs():
    list_of_inputs = []
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([1.5, 2.7, 3.2]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([-0.5, -1.5, -2.5]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[[ -1.0, -2.0], [-3.0, -4.0]], [[-5.0, -6.0], [-7.0, -8.0]]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([1.0]).numpy()
    list_of_inputs.append({"input": input})
    
    return list_of_inputs

generated_inputs["torch.floor_"] = floor__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_'.")


check_valid('torch.floor_', generated_inputs['torch.floor_'], lib="torch", suffix=0)
