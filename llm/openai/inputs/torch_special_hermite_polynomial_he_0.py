
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hermite_polynomial_he_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    n1 = 2
    input_dict1 = {'x': input1, 'n': n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, -2.0, -3.0])
    n2 = 3
    input_dict2 = {'x': input2, 'n': n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.0, 0.0, 0.0])
    n3 = 0
    input_dict3 = {'x': input3, 'n': n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0])
    n4 = 5
    input_dict4 = {'x': input4, 'n': n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    n5 = 1
    input_dict5 = {'x': input5, 'n': n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    n6 = 4
    input_dict6 = {'x': input6, 'n': n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.5, 1.5, 2.5])
    n7 = 2
    input_dict7 = {'x': input7, 'n': n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1.0, 2.0, 3.0, 4.0])
    n8 = 0
    input_dict8 = {'x': input8, 'n': n8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    n9 = 3
    input_dict9 = {'x': input9, 'n': n9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[-0.5, 0.5], [1.5, -1.5]])
    n10 = 1
    input_dict10 = {'x': input10, 'n': n10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.special.hermite_polynomial_he"] = hermite_polynomial_he_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.hermite_polynomial_he' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.hermite_polynomial_he'.")


check_valid('torch.special.hermite_polynomial_he', generated_inputs['torch.special.hermite_polynomial_he'], lib="torch", suffix=0)
