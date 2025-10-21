
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_asarray_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3], dtype=np.float32)
    dtype1 = torch.float32
    copy1 = True
    requires_grad1 = False
    input_dict1 = {'obj': input1, 'dtype': dtype1, 'copy': copy1, 'requires_grad': requires_grad1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dtype2 = torch.float32
    copy2 = False
    requires_grad2 = True
    input_dict2 = {'obj': input2, 'dtype': dtype2, 'copy': copy2, 'requires_grad': requires_grad2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, -2.5, 3.7], dtype=np.float32)
    dtype3 = torch.float32
    copy3 = True
    requires_grad3 = True
    input_dict3 = {'obj': input3, 'dtype': dtype3, 'copy': copy3, 'requires_grad': requires_grad3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([], dtype=np.float32)
    dtype4 = torch.float32
    copy4 = False
    requires_grad4 = False
    input_dict4 = {'obj': input4, 'dtype': dtype4, 'copy': copy4, 'requires_grad': requires_grad4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.asarray"] = torch_asarray_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.asarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asarray'.")


check_valid('torch.asarray', generated_inputs['torch.asarray'], lib="torch", suffix=0)
