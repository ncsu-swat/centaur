
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []
    input_dict = {}
    
    input1 = torch.randn(1, 3, 32, 32)
    input_dict["kernel_size"] = 2
    input_dict["stride"] = (1, 1)
    input_dict["padding"] = (0, 0)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = False
    input_dict["ceil_mode"] = False
    input_dict["input"] = input1.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input2 = torch.randn(1, 1, 64, 64)
    input_dict["kernel_size"] = (3, 3)
    input_dict["stride"] = (2, 2)
    input_dict["padding"] = (1, 1)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = True
    input_dict["ceil_mode"] = False
    input_dict["input"] = input2.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input3 = torch.randn(2, 5, 128, 128)
    input_dict["kernel_size"] = 4
    input_dict["stride"] = (4, 4)
    input_dict["padding"] = (2, 2)
    input_dict["dilation"] = (2, 2)
    input_dict["return_indices"] = False
    input_dict["ceil_mode"] = True
    input_dict["input"] = input3.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input4 = torch.randn(1, 8, 256, 256)
    input_dict["kernel_size"] = (2, 4)
    input_dict["stride"] = (1, 2)
    input_dict["padding"] = (0, 1)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = True
    input_dict["ceil_mode"] = False
    input_dict["input"] = input4.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input5 = torch.randn(4, 2, 32, 32)
    input_dict["kernel_size"] = 1
    input_dict["stride"] = (1, 1)
    input_dict["padding"] = (0, 0)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = False
    input_dict["ceil_mode"] = True
    input_dict["input"] = input5.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input6 = torch.randn(1, 3, 64, 64)
    input_dict["kernel_size"] = (3, 3)
    input_dict["stride"] = (2, 2)
    input_dict["padding"] = (1, 1)
    input_dict["dilation"] = (2, 2)
    input_dict["return_indices"] = True
    input_dict["ceil_mode"] = True
    input_dict["input"] = input6.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input7 = torch.randn(2, 5, 128, 128)
    input_dict["kernel_size"] = 4
    input_dict["stride"] = (4, 4)
    input_dict["padding"] = (2, 2)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = False
    input_dict["ceil_mode"] = False
    input_dict["input"] = input7.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input8 = torch.randn(1, 8, 256, 256)
    input_dict["kernel_size"] = (2, 4)
    input_dict["stride"] = (1, 2)
    input_dict["padding"] = (0, 1)
    input_dict["dilation"] = (2, 2)
    input_dict["return_indices"] = True
    input_dict["ceil_mode"] = True
    input_dict["input"] = input8.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input9 = torch.randn(4, 2, 32, 32)
    input_dict["kernel_size"] = 1
    input_dict["stride"] = (1, 1)
    input_dict["padding"] = (0, 0)
    input_dict["dilation"] = (2, 2)
    input_dict["return_indices"] = False
    input_dict["ceil_mode"] = False
    input_dict["input"] = input9.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input10 = torch.randn(1, 3, 64, 64)
    input_dict["kernel_size"] = (5, 5)
    input_dict["stride"] = (3, 3)
    input_dict["padding"] = (2, 2)
    input_dict["dilation"] = (1, 1)
    input_dict["return_indices"] = True
    input_dict["ceil_mode"] = True
    input_dict["input"] = input10.numpy()
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_3"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_3'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_3'], lib="torch", suffix=3)
