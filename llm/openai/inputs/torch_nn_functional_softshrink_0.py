
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softshrink_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, -2.0, 3.0, -4.0])
    lambd1 = 0.5
    input_dict1 = {"input": input1, "lambd": lambd1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [-3.0, 4.0]])
    lambd2 = 0.1
    input_dict2 = {"input": input2, "lambd": lambd2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.0, 0.0, 0.0])
    lambd3 = 1.0
    input_dict3 = {"input": input3, "lambd": lambd3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([-1.0, -2.0, -3.0])
    lambd4 = 0.8
    input_dict4 = {"input": input4, "lambd": lambd4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, -2.5, 3.5])
    lambd5 = 0.2
    input_dict5 = {"input": input5, "lambd": lambd5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[[1.0], [2.0]], [[-3.0], [4.0]]])
    lambd6 = 0.7
    input_dict6 = {"input": input6, "lambd": lambd6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([10.0])
    lambd7 = 0.9
    input_dict7 = {"input": input7, "lambd": lambd7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([-1.1, 2.2, -3.3, 4.4])
    lambd8 = 0.3
    input_dict8 = {"input": input8, "lambd": lambd8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    lambd9 = 0.6
    input_dict9 = {"input": input9, "lambd": lambd9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lambd10 = 0.4
    input_dict10 = {"input": input10, "lambd": lambd10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softshrink'.")


check_valid('torch.nn.functional.softshrink', generated_inputs['torch.nn.functional.softshrink'], lib="torch", suffix=0)
