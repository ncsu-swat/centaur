
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def select_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3, 4, 5])
    dim1 = 0
    index1 = 2
    input_dict1 = {"input": input1, "dim": dim1, "index": index1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dim2 = 1
    index2 = 0
    input_dict2 = {"input": input2, "dim": dim2, "index": index2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dim3 = 2
    index3 = 1
    input_dict3 = {"input": input3, "dim": dim3, "index": index3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 2, 3, 4])
    dim4 = 0
    index4 = -1
    input_dict4 = {"input": input4, "dim": dim4, "index": index4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 2], [3, 4]])
    dim5 = 1
    index5 = -1
    input_dict5 = {"input": input5, "dim": dim5, "index": index5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dim6 = 0
    index6 = 1
    input_dict6 = {"input": input6, "dim": dim6, "index": index6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5, 6])
    dim7 = 0
    index7 = 0
    input_dict7 = {"input": input7, "dim": dim7, "index": index7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    dim8 = 1
    index8 = 3
    input_dict8 = {"input": input8, "dim": dim8, "index": index8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    dim9 = 2
    index9 = 0
    input_dict9 = {"input": input9, "dim": dim9, "index": index9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([10, 20, 30, 40, 50])
    dim10 = 0
    index10 = 4
    input_dict10 = {"input": input10, "dim": dim10, "index": index10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.select"] = select_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.select'.")


check_valid('torch.select', generated_inputs['torch.select'], lib="torch", suffix=0)
