
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_dist_inputs():
    list_of_inputs = []

    input1 = np.array([-1.0, 2.0, -3.0])
    other1 = np.array([4.0, -5.0, 6.0])
    p1 = 2.0
    list_of_inputs.append({"input": input1, "other": other1, "p": p1})

    input2 = np.array([1.0, 1.0])
    other2 = np.array([2.0, 2.0])
    p2 = 1.0
    list_of_inputs.append({"input": input2, "other": other2, "p": p2})

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p3 = 3.0
    list_of_inputs.append({"input": input3, "other": other3, "p": p3})

    input4 = np.array([0.0, 0.0, 0.0])
    other4 = np.array([1.0, 2.0, 3.0])
    p4 = 0.5
    list_of_inputs.append({"input": input4, "other": other4, "p": p4})

    input5 = np.array([-1.0, -2.0, -3.0])
    other5 = np.array([1.0, 2.0, 3.0])
    p5 = 4.0
    list_of_inputs.append({"input": input5, "other": other5, "p": p5})

    input6 = np.array([1.0])
    other6 = np.array([2.0])
    p6 = 1.5
    list_of_inputs.append({"input": input6, "other": other6, "p": p6})

    input7 = np.array([1.0, 2.0, 3.0, 4.0])
    other7 = np.array([5.0, 6.0, 7.0, 8.0])
    p7 = 0.0
    list_of_inputs.append({"input": input7, "other": other7, "p": p7})

    input8 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other8 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    p8 = 2.5
    list_of_inputs.append({"input": input8, "other": other8, "p": p8})
    
    input9 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    other9 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    p9 = 1.0
    list_of_inputs.append({"input": input9, "other": other9, "p": p9})
    
    input10 = np.array([1.5, -2.5, 3.5])
    other10 = np.array([-0.5, 1.5, -2.5])
    p10 = 3.0
    list_of_inputs.append({"input": input10, "other": other10, "p": p10})

    return list_of_inputs

generated_inputs["torch.dist"] = torch_dist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dist'.")


check_valid('torch.dist', generated_inputs['torch.dist'], lib="torch", suffix=0)
