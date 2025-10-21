
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ravel_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input1)}})
    
    input2 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input2)}})
    
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input3)}})
    
    input4 = np.array([1, -2, 3, -4])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input4)}})
    
    input5 = np.random.rand(2, 3, 4)
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input5)}})
    
    input6 = np.zeros((5, 5))
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input6)}})
    
    input7 = np.ones((1, 2, 3, 4))
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input7)}})
    
    input8 = np.array([[[[1]]]])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input8)}})

    input9 = np.random.randint(0, 10, (2, 2, 2))
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input9)}})

    input10 = np.array([1.0, 2.5, 3.7])
    list_of_inputs.append({"torch.ravel": {"input": torch.tensor(input10)}})
    
    return list_of_inputs

generated_inputs["torch.ravel"] = ravel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ravel'.")


check_valid('torch.ravel', generated_inputs['torch.ravel'], lib="torch", suffix=0)
