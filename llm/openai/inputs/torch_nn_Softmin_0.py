
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softmin_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3)
    dim1 = 1
    input_dict1 = {"dim": dim1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(5)
    dim2 = 0
    input_dict2 = {"dim": dim2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.randn(3, 3, 3)
    dim3 = 2
    input_dict3 = {"dim": dim3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 2)
    dim4 = 1
    input_dict4 = {"dim": dim4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(4, 5, 6, 7)
    dim5 = 3
    input_dict5 = {"dim": dim5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 3)
    dim6 = 0
    input_dict6 = {"dim": dim6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(1, 4)
    dim7 = 1
    input_dict7 = {"dim": dim7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 2, 4)
    dim8 = 0
    input_dict8 = {"dim": dim8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(5, 5)
    dim9 = 1
    input_dict9 = {"dim": dim9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.randn(2, 4, 5)
    dim10 = 2
    input_dict10 = {"dim": dim10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmin'.")


check_valid('torch.nn.Softmin', generated_inputs['torch.nn.Softmin'], lib="torch", suffix=0)
