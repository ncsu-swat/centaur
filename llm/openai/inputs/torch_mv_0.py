
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_mv_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy().astype(np.float32)
    vec1 = torch.randn(3).numpy().astype(np.float32)
    out1 = torch.tensor(np.array([])).numpy().astype(np.float32)
    input_dict1 = {"input": input1, "vec": vec1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(4, 2).numpy().astype(np.float32)
    vec2 = torch.randn(2).numpy().astype(np.float32)
    out2 = torch.tensor(np.array([])).numpy().astype(np.float32)
    input_dict2 = {"input": input2, "vec": vec2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 5).numpy().astype(np.float32)
    vec3 = torch.randn(5).numpy().astype(np.float32)
    out3 = torch.tensor(np.array([])).numpy().astype(np.float32)
    input_dict3 = {"input": input3, "vec": vec3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 7).numpy().astype(np.float32)
    vec4 = torch.randn(7).numpy().astype(np.float32)
    out4 = torch.tensor(np.array([])).numpy().astype(np.float32)
    input_dict4 = {"input": input4, "vec": vec4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 1).numpy().astype(np.float32)
    vec5 = torch.randn(1).numpy().astype(np.float32)
    out5 = torch.tensor(np.array([])).numpy().astype(np.float32)
    input_dict5 = {"input": input5, "vec": vec5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.mv"] = torch_mv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mv'.")


check_valid('torch.mv', generated_inputs['torch.mv'], lib="torch", suffix=0)
