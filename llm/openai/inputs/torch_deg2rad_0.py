
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def deg2rad_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.0, 30.0, 45.0, 60.0])
    out1 = np.empty_like(input1)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.0, 90.0], [180.0, 270.0]])
    out2 = np.empty_like(input2)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-45.0, -90.0], [0.0, 45.0]])
    out3 = np.empty_like(input3)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0, 4.0])
    out4 = np.empty_like(input4)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-180.0, 180.0], [360.0, -360.0]])
    out5 = np.empty_like(input5)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([10.0])
    out6 = np.empty_like(input6)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[0.0, 45.0]], [[90.0, 135.0]]])
    out7 = np.empty_like(input7)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0.0, 0.0, 0.0])
    out8 = np.empty_like(input8)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([179.9, -179.9])
    out9 = np.empty_like(input9)
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out10 = np.empty_like(input10)
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.deg2rad"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.deg2rad'.")


check_valid('torch.deg2rad', generated_inputs['torch.deg2rad'], lib="torch", suffix=0)
