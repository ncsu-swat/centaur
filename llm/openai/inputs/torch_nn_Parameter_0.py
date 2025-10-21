
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nn_parameter_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    requires_grad1 = True
    input_dict1 = {"data": input1, "requires_grad": requires_grad1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    requires_grad2 = False
    input_dict2 = {"data": input2, "requires_grad": requires_grad2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    requires_grad3 = True
    input_dict3 = {"data": input3, "requires_grad": requires_grad3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3, 4).astype(np.float16)
    requires_grad4 = False
    input_dict4 = {"data": input4, "requires_grad": requires_grad4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.zeros((5,), dtype=np.float32)
    requires_grad5 = True
    input_dict5 = {"data": input5, "requires_grad": requires_grad5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.ones((1, 2, 3, 4), dtype=np.float64)
    requires_grad6 = False
    input_dict6 = {"data": input6, "requires_grad": requires_grad6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    requires_grad7 = True
    input_dict7 = {"data": input7, "requires_grad": requires_grad7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([100.0], dtype=np.float64)
    requires_grad8 = False
    input_dict8 = {"data": input8, "requires_grad": requires_grad8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.randn(3, 3).astype(np.float16)
    requires_grad9 = True
    input_dict9 = {"data": input9, "requires_grad": requires_grad9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    requires_grad10 = False
    input_dict10 = {"data": input10, "requires_grad": requires_grad10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.Parameter"] = nn_parameter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Parameter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Parameter'.")


check_valid('torch.nn.Parameter', generated_inputs['torch.nn.Parameter'], lib="torch", suffix=0)
