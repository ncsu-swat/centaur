
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def polygamma_inputs():
    list_of_inputs = []
    
    n = 0
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 1
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 2
    input = torch.randn(2, 2).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    n = 3
    input = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    n = 4
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    n = 5
    input = torch.tensor([[-1.0], [0.0], [1.0]]).numpy()
    out = np.empty((0,))
    input_dict = {"n": n, "input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.polygamma"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.polygamma'.")


check_valid('torch.polygamma', generated_inputs['torch.polygamma'], lib="torch", suffix=0)
