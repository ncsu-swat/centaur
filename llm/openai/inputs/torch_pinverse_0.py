
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pinverse_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    rcond1 = 1e-10
    input_dict1 = {"input": input1, "rcond": rcond1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rcond2 = 1e-15
    input_dict2 = {"input": input2, "rcond": rcond2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1.0, 0.0], [0.0, 1.0]])
    rcond3 = 1e-08
    input_dict3 = {"input": input3, "rcond": rcond3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 2.0], [2.0, 4.0]])
    rcond4 = 1e-06
    input_dict4 = {"input": input4, "rcond": rcond4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.0], [2.0], [3.0]])
    rcond5 = 1e-12
    input_dict5 = {"input": input5, "rcond": rcond5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[1.0, 2.0, 3.0]])
    rcond6 = 1e-13
    input_dict6 = {"input": input6, "rcond": rcond6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    rcond7 = 1e-11
    input_dict7 = {"input": input7, "rcond": rcond7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[0.0, 0.0], [0.0, 0.0]])
    rcond8 = 1e-14
    input_dict8 = {"input": input8, "rcond": rcond8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rcond9 = 1e-09
    input_dict9 = {"input": input9, "rcond": rcond9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[2.0, 1.0], [1.0, 2.0]])
    rcond10 = 1e-07
    input_dict10 = {"input": input10, "rcond": rcond10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.pinverse"] = pinverse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pinverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pinverse'.")


check_valid('torch.pinverse', generated_inputs['torch.pinverse'], lib="torch", suffix=0)
