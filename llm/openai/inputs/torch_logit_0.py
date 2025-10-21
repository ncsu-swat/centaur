
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logit_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.1, 0.2, 0.3])
    eps1 = 1e-6
    out1 = np.zeros_like(input1)
    input_dict1 = {"input": input1, "eps": eps1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([0.5, 0.7, 0.9])
    eps2 = 1e-8
    out2 = np.array([0.0, 0.0, 0.0])
    input_dict2 = {"input": input2, "eps": eps2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-0.1, -0.2, -0.3])
    eps3 = 1e-7
    out3 = np.empty_like(input3)
    input_dict3 = {"input": input3, "eps": eps3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.99, 0.98, 0.97])
    eps4 = 1e-5
    out4 = np.array([0.0, 0.0, 0.0])
    input_dict4 = {"input": input4, "eps": eps4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[0.1, 0.2], [0.3, 0.4]])
    eps5 = 1e-9
    out5 = np.zeros_like(input5)
    input_dict5 = {"input": input5, "eps": eps5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[0.5, 0.6], [0.7, 0.8]])
    eps6 = 1e-4
    out6 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict6 = {"input": input6, "eps": eps6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-0.1, -0.2], [-0.3, -0.4]])
    eps7 = 1e-6
    out7 = np.empty_like(input7)
    input_dict7 = {"input": input7, "eps": eps7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([0.9, 0.8, 0.7, 0.6])
    eps8 = 1e-7
    out8 = np.array([0.0, 0.0, 0.0, 0.0])
    input_dict8 = {"input": input8, "eps": eps8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(2, 2, 2)
    eps9 = 1e-8
    out9 = np.zeros_like(input9)
    input_dict9 = {"input": input9, "eps": eps9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    eps10 = 1e-5
    out10 = np.empty_like(input10)
    input_dict10 = {"input": input10, "eps": eps10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.logit"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logit'.")


check_valid('torch.logit', generated_inputs['torch.logit'], lib="torch", suffix=0)
