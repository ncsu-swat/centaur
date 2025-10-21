
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pad_inputs():
    list_of_inputs = []
    input1 = np.random.rand(3, 3, 4, 2).astype(np.float32)
    pad1 = (1, 1)
    mode1 = 'constant'
    value1 = 0.0
    input_dict1 = {'input': input1, 'pad': pad1, 'mode': mode1, 'value': value1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 3, 4, 2).astype(np.float32)
    pad2 = (1, 1, 2, 2)
    mode2 = 'constant'
    value2 = 1.0
    input_dict2 = {'input': input2, 'pad': pad2, 'mode': mode2, 'value': value2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(3, 3, 4, 2).astype(np.float32)
    pad3 = (0, 1, 2, 1, 3, 3)
    mode3 = 'constant'
    value3 = -1.0
    input_dict3 = {'input': input3, 'pad': pad3, 'mode': mode3, 'value': value3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3).astype(np.float32)
    pad4 = (2, 2)
    mode4 = 'constant'
    value4 = 0.5
    input_dict4 = {'input': input4, 'pad': pad4, 'mode': mode4, 'value': value4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(4, 5, 6).astype(np.float32)
    pad5 = (1, 1, 1, 1)
    mode5 = 'replicate'
    input_dict5 = {'input': input5, 'pad': pad5, 'mode': mode5, 'value': 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    pad6 = (0, 1, 0, 1)
    mode6 = 'constant'
    value6 = -0.5
    input_dict6 = {'input': input6, 'pad': pad6, 'mode': mode6, 'value': value6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(5, 5).astype(np.float32)
    pad7 = (3, 3)
    mode7 = 'constant'
    value7 = 10.0
    input_dict7 = {'input': input7, 'pad': pad7, 'mode': mode7, 'value': value7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.functional.pad"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pad'.")


check_valid('torch.nn.functional.pad', generated_inputs['torch.nn.functional.pad'], lib="torch", suffix=0)
