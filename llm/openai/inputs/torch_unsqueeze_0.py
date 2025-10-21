
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unsqueeze_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3, 4])
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([1, 2, 3, 4])
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1, 2], [3, 4]])
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1, 2], [3, 4]])
    dim4 = 1
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dim5 = 0
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dim6 = 2
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1, 2, 3])
    dim7 = -1
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 2], [3, 4]])
    dim8 = -2
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.unsqueeze"] = unsqueeze_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unsqueeze' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unsqueeze'.")


check_valid('torch.unsqueeze', generated_inputs['torch.unsqueeze'], lib="torch", suffix=0)
