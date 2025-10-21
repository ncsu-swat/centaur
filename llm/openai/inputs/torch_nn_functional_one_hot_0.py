
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def one_hot_inputs():
    list_of_inputs = []
    
    input1 = np.array([0, 1, 2, 3])
    num_classes1 = 5
    input_dict1 = {"input": input1, "num_classes": num_classes1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([2, 0, 1, 0, 2])
    num_classes2 = 3
    input_dict2 = {"input": input2, "num_classes": num_classes2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0, 1, 2])
    num_classes3 = 4
    input_dict3 = {"input": input3, "num_classes": num_classes3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0, 1], [2, 3]], dtype=np.int64)
    num_classes4 = 5
    input_dict4 = {"input": input4, "num_classes": num_classes4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0, 1, 2, 0, 1, 2])
    num_classes5 = 3
    input_dict5 = {"input": input5, "num_classes": num_classes5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([5, 6, 7, 8])
    num_classes6 = 10
    input_dict6 = {"input": input6, "num_classes": num_classes6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[0, 1, 2], [3, 4, 5]])
    num_classes7 = 6
    input_dict7 = {"input": input7, "num_classes": num_classes7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0, 0, 0, 0])
    num_classes8 = 1
    input_dict8 = {"input": input8, "num_classes": num_classes8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.nn.functional.one_hot"] = one_hot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.one_hot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.one_hot'.")


check_valid('torch.nn.functional.one_hot', generated_inputs['torch.nn.functional.one_hot'], lib="torch", suffix=0)
