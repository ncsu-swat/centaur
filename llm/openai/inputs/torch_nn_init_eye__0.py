
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def eye_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    input_dict1 = {"tensor": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 0], [0, 1]], dtype=np.float64)
    input_dict2 = {"tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float16)
    input_dict3 = {"tensor": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    input_dict4 = {"tensor": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=np.float64)
    input_dict5 = {"tensor": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    input_dict6 = {"tensor": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1]], dtype=np.float64)
    input_dict7 = {"tensor": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float16)
    input_dict8 = {"tensor": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    input_dict9 = {"tensor": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 0, 0, 0, 0]], dtype=np.float64)
    input_dict10 = {"tensor": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.init.eye_"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.eye_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.eye_'.")


check_valid('torch.nn.init.eye_', generated_inputs['torch.nn.init.eye_'], lib="torch", suffix=0)
