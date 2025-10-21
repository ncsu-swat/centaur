
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2)
    alpha1 = 1.0
    inplace1 = False
    input_dict1 = {"alpha": alpha1, "inplace": inplace1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(3, 4)
    alpha2 = 2.5
    inplace2 = True
    input_dict2 = {"alpha": alpha2, "inplace": inplace2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.zeros((5, 5, 5))
    alpha3 = 0.5
    inplace3 = False
    input_dict3 = {"alpha": alpha3, "inplace": inplace3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.ones((2, 2))
    alpha4 = 1.5
    inplace4 = True
    input_dict4 = {"alpha": alpha4, "inplace": inplace4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(100)
    alpha5 = 0.1
    inplace5 = False
    input_dict5 = {"alpha": alpha5, "inplace": inplace5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.full((4, 3), -1.0)
    alpha6 = 3.0
    inplace6 = True
    input_dict6 = {"alpha": alpha6, "inplace": inplace6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 3, 4, 5)
    alpha7 = 0.75
    inplace7 = False
    input_dict7 = {"alpha": alpha7, "inplace": inplace7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-2.0, 1.5], [0.0, -3.2]])
    alpha8 = 1.2
    inplace8 = True
    input_dict8 = {"alpha": alpha8, "inplace": inplace8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(6, 7)
    alpha9 = 2.0
    inplace9 = False
    input_dict9 = {"alpha": alpha9, "inplace": inplace9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.empty((1, 1, 1))
    alpha10 = 0.2
    inplace10 = True
    input_dict10 = {"alpha": alpha10, "inplace": inplace10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.CELU"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CELU'.")


check_valid('torch.nn.CELU', generated_inputs['torch.nn.CELU'], lib="torch", suffix=0)
