
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def inverse_inputs():
    list_of_inputs = []

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out1 = np.empty((2, 2), dtype=np.float32)
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=np.float64)
    out2 = np.empty((3, 3), dtype=np.float64)
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out3 = np.empty((3, 3), dtype=np.float32)
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    out4 = np.empty((2, 2), dtype=np.float64)
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[0.5, 0.2], [0.1, 0.8]], dtype=np.float32)
    out5 = np.empty((2, 2), dtype=np.float32)
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float64)
    out6 = np.empty((3, 3), dtype=np.float64)
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, 1.0], [1.0, 1.1]], dtype=np.float32)
    out7 = np.empty((2, 2), dtype=np.float32)
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[5.0, 3.0, 1.0], [2.0, 4.0, 7.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    out8 = np.empty((3, 3), dtype=np.float64)
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.inverse"] = inverse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.inverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inverse'.")


check_valid('torch.inverse', generated_inputs['torch.inverse'], lib="torch", suffix=0)
