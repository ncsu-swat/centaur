
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def softplus_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    beta1 = 1.0
    threshold1 = 20.0
    input_dict1 = {"input": input1, "beta": beta1, "threshold": threshold1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    beta2 = 2.0
    threshold2 = 10.0
    input_dict2 = {"input": input2, "beta": beta2, "threshold": threshold2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    beta3 = 0.5
    threshold3 = 5.0
    input_dict3 = {"input": input3, "beta": beta3, "threshold": threshold3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    beta4 = 1.0
    threshold4 = 25.0
    input_dict4 = {"input": input4, "beta": beta4, "threshold": threshold4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([], dtype=np.float32)
    beta5 = 1.5
    threshold5 = 15.0
    input_dict5 = {"input": input5, "beta": beta5, "threshold": threshold5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta6 = 0.1
    threshold6 = 1.0
    input_dict6 = {"input": input6, "beta": beta6, "threshold": threshold6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0], dtype=np.float32)
    beta7 = 5.0
    threshold7 = 2.0
    input_dict7 = {"input": input7, "beta": beta7, "threshold": threshold7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    beta8 = 1.0
    threshold8 = 10.0
    input_dict8 = {"input": input8, "beta": beta8, "threshold": threshold8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    beta9 = 1.0
    threshold9 = 20.0
    input_dict9 = {"input": input9, "beta": beta9, "threshold": threshold9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    beta10 = 3.0
    threshold10 = 10.0
    input_dict10 = {"input": input10, "beta": beta10, "threshold": threshold10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["torch.nn.functional.softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softplus'.")


check_valid('torch.nn.functional.softplus', generated_inputs['torch.nn.functional.softplus'], lib="torch", suffix=0)
