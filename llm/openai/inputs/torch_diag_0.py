
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def diag_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    diagonal1 = 0
    out1 = np.array([])
    input_dict1 = {"input": input1, "diagonal": diagonal1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    diagonal2 = 0
    out2 = np.array([])
    input_dict2 = {"input": input2, "diagonal": diagonal2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 2.0, 3.0])
    diagonal3 = 1
    out3 = np.array([])
    input_dict3 = {"input": input3, "diagonal": diagonal3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    diagonal4 = -1
    out4 = np.array([])
    input_dict4 = {"input": input4, "diagonal": diagonal4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0, 2.0, 3.0, 4.0])
    diagonal5 = 0
    out5 = np.array([])
    input_dict5 = {"input": input5, "diagonal": diagonal5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    diagonal6 = 1
    out6 = np.array([])
    input_dict6 = {"input": input6, "diagonal": diagonal6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    diagonal7 = -1
    out7 = np.array([])
    input_dict7 = {"input": input7, "diagonal": diagonal7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0])
    diagonal8 = 0
    out8 = np.array([])
    input_dict8 = {"input": input8, "diagonal": diagonal8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    diagonal9 = 2
    out9 = np.array([])
    input_dict9 = {"input": input9, "diagonal": diagonal9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0], [3.0, 4.0]])
    diagonal10 = -1
    out10 = np.array([])
    input_dict10 = {"input": input10, "diagonal": diagonal10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.diag"] = diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag'.")


check_valid('torch.diag', generated_inputs['torch.diag'], lib="torch", suffix=0)
