
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def eigvalsh_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 2)
    uplo1 = 'L'
    out1 = np.zeros((2,))
    input_dict1 = {'A': input1, 'UPLO': uplo1, 'out': out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 3)
    uplo2 = 'U'
    out2 = np.zeros((3,))
    input_dict2 = {'A': input2, 'UPLO': uplo2, 'out': out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(4, 4)
    uplo3 = 'L'
    out3 = np.zeros((4,))
    input_dict3 = {'A': input3, 'UPLO': uplo3, 'out': out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 2, 2, 2)
    uplo4 = 'U'
    out4 = np.zeros((2, 2))
    input_dict4 = {'A': input4, 'UPLO': uplo4, 'out': out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    uplo5 = 'L'
    out5 = np.zeros((3,), dtype=np.float64)
    input_dict5 = {'A': input5, 'UPLO': uplo5, 'out': out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 2)
    uplo6 = 'L'
    out6 = np.array([])
    input_dict6 = {'A': input6, 'UPLO': uplo6, 'out': out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4)
    uplo7 = 'U'
    out7 = np.zeros((4,))
    input_dict7 = {'A': input7, 'UPLO': uplo7, 'out': out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2) - 0.5
    uplo8 = 'L'
    out8 = np.zeros((2,))
    input_dict8 = {'A': input8, 'UPLO': uplo8, 'out': out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(3, 3)
    uplo9 = 'U'
    out9 = np.zeros((3,))
    input_dict9 = {'A': input9, 'UPLO': uplo9, 'out': out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 2)
    uplo10 = 'L'
    out10 = np.zeros((2,))
    input_dict10 = {'A': input10, 'UPLO': uplo10, 'out': out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.linalg.eigvalsh"] = eigvalsh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.eigvalsh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigvalsh'.")


check_valid('torch.linalg.eigvalsh', generated_inputs['torch.linalg.eigvalsh'], lib="torch", suffix=0)
