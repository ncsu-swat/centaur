
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def argmin_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    dim1 = 0
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim2 = 1
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    dim3 = 0
    keepdim3 = False
    input_dict3 = {"input": input3, "dim": dim3, "keepdim": keepdim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(3, 4, 5)
    dim4 = 2
    keepdim4 = True
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([5.0])
    dim5 = None
    keepdim5 = False
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[1.0, -2.0], [3.0, 4.0]])
    dim6 = 1
    keepdim6 = False
    input_dict6 = {"input": input6, "dim": dim6, "keepdim": keepdim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    dim7 = 0
    keepdim7 = True
    input_dict7 = {"input": input7, "dim": dim7, "keepdim": keepdim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([0.0, 0.0, 0.0])
    dim8 = None
    keepdim8 = False
    input_dict8 = {"input": input8, "dim": dim8, "keepdim": keepdim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[-10.0, 5.0], [2.0, -8.0]])
    dim9 = 1
    keepdim9 = False
    input_dict9 = {"input": input9, "dim": dim9, "keepdim": keepdim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 2, 2, 2)
    dim10 = 3
    keepdim10 = True
    input_dict10 = {"input": input10, "dim": dim10, "keepdim": keepdim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmin'.")


check_valid('torch.argmin', generated_inputs['torch.argmin'], lib="torch", suffix=0)
