
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def i0e_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0])
    input_dict1 = {"input": torch.tensor(input1).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([0.0])
    input_dict2 = {"input": torch.tensor(input2).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-1.0])
    input_dict3 = {"input": torch.tensor(input3).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0])
    input_dict4 = {"input": torch.tensor(input4).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, -2.0, 3.0])
    input_dict5 = {"input": torch.tensor(input5).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict6 = {"input": torch.tensor(input6).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict7 = {"input": torch.tensor(input7).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.5, 2.5, 3.5])
    input_dict8 = {"input": torch.tensor(input8).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.0], [2.0], [3.0]])
    input_dict9 = {"input": torch.tensor(input9).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100.0])
    input_dict10 = {"input": torch.tensor(input10).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0e'.")


check_valid('torch.special.i0e', generated_inputs['torch.special.i0e'], lib="torch", suffix=0)
