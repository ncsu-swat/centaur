
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad2d_inputs():
    list_of_inputs = []

    input1 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    padding1 = (2, 2, 2, 2)
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    padding2 = (1, 1, 0, 1)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    padding3 = (1,1,1,1)
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    padding4 = (0, 1, 2, 3)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
    padding5 = (0, 0, 1, 0)
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.ones((2, 3, 4, 5)).astype(np.float32)
    padding6 = (1,1,1,1)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.arange(49, dtype=np.float32).reshape(1, 1, 7, 7)
    padding7 = (2, 0, 1, 0)
    input_dict7 = {"padding": padding7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad2d_2"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_2'.")


check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_2'], lib="torch", suffix=2)
