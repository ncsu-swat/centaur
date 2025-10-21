
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def vecdot_inputs():
    list_of_inputs = []
    
    x1 = torch.randn(3, 2).numpy()
    y1 = torch.randn(3, 2).numpy()
    dim1 = -1
    out1 = torch.zeros((3,), dtype=torch.float32).numpy()
    input_dict1 = {"x": x1, "y": y1, "dim": dim1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    x2 = torch.randn(2, 4).numpy()
    y2 = torch.randn(2, 4).numpy()
    dim2 = 1
    out2 = torch.zeros((2,), dtype=torch.float32).numpy()
    input_dict2 = {"x": x2, "y": y2, "dim": dim2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    x3 = torch.randn(5, 3).numpy()
    y3 = torch.randn(5, 3).numpy()
    dim3 = -1
    out3 = torch.zeros((5,), dtype=torch.float32).numpy()
    input_dict3 = {"x": x3, "y": y3, "dim": dim3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    x4 = torch.randn(1, 1).numpy()
    y4 = torch.randn(1, 1).numpy()
    dim4 = 0
    out4 = torch.zeros((1,), dtype=torch.float32).numpy()
    input_dict4 = {"x": x4, "y": y4, "dim": dim4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    x5 = torch.randn(4, 2, 3).numpy()
    y5 = torch.randn(4, 2, 3).numpy()
    dim5 = 2
    out5 = torch.zeros((4, 2), dtype=torch.float32).numpy()
    input_dict5 = {"x": x5, "y": y5, "dim": dim5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    x6 = torch.randn(2, 3).numpy()
    y6 = torch.randn(2, 3).numpy()
    dim6 = -1
    out6 = torch.zeros((2,), dtype=torch.float32).numpy()
    input_dict6 = {"x": x6, "y": y6, "dim": dim6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    x7 = torch.randn(5, 1).numpy()
    y7 = torch.randn(5, 1).numpy()
    dim7 = 0
    out7 = torch.zeros((5,), dtype=torch.float32).numpy()
    input_dict7 = {"x": x7, "y": y7, "dim": dim7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    x8 = torch.randn(3, 2, 4).numpy()
    y8 = torch.randn(3, 2, 4).numpy()
    dim8 = -1
    out8 = torch.zeros((3, 2), dtype=torch.float32).numpy()
    input_dict8 = {"x": x8, "y": y8, "dim": dim8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    x9 = torch.randn(2, 2).numpy()
    y9 = torch.randn(2, 2).numpy()
    dim9 = 1
    out9 = torch.zeros((2,), dtype=torch.float32).numpy()
    input_dict9 = {"x": x9, "y": y9, "dim": dim9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.linalg.vecdot"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.vecdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vecdot'.")


check_valid('torch.linalg.vecdot', generated_inputs['torch.linalg.vecdot'], lib="torch", suffix=0)
