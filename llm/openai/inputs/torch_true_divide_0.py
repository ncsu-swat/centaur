
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def true_divide_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    other1 = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32).numpy()
    out1 = np.array([])
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).numpy()
    other2 = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32).numpy()
    out2 = np.array([])
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32).numpy()
    other3 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    out3 = np.array([])
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.true_divide"] = true_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.true_divide'.")


check_valid('torch.true_divide', generated_inputs['torch.true_divide'], lib="torch", suffix=0)
