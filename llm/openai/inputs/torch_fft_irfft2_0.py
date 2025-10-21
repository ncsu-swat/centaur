
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def irfft2_inputs():
    list_of_inputs = []

    input1 = np.random.rand(2, 3) + 1j * np.random.rand(2, 3)
    s1 = (4, 6)
    dim1 = (-2, -1)
    norm1 = "backward"
    out1 = np.zeros((2, 3), dtype=np.float32)
    input_dict1 = {"input": input1, "s": s1, "dim": dim1, "norm": norm1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, 4, 2) + 1j * np.random.rand(4, 4, 2)
    s2 = (8, 8, 2)
    dim2 = (-3, -2, -1)
    norm2 = "ortho"
    out2 = np.zeros((4, 4, 2), dtype=np.float32)
    input_dict2 = {"input": input2, "s": s2, "dim": dim2, "norm": norm2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    s3 = (2, 2)
    dim3 = (-2, -1)
    norm3 = "forward"
    out3 = np.zeros((2, 2), dtype=np.float32)
    input_dict3 = {"input": input3, "s": s3, "dim": dim3, "norm": norm3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 5) + 1j * np.random.rand(1, 5)
    s4 = (1, 6)
    dim4 = (-2, -1)
    norm4 = "backward"
    out4 = np.zeros((1, 5), dtype=np.float32)
    input_dict4 = {"input": input4, "s": s4, "dim": dim4, "norm": norm4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    s5 = (3, 3)
    dim5 = (-2, -1)
    norm5 = "ortho"
    out5 = np.zeros((3, 3), dtype=np.float32)
    input_dict5 = {"input": input5, "s": s5, "dim": dim5, "norm": norm5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(5, 2, 4) + 1j * np.random.rand(5, 2, 4)
    s6 = (5, 4, 8)
    dim6 = (-3, -2, -1)
    norm6 = "backward"
    out6 = np.zeros((5, 2, 4), dtype=np.float32)
    input_dict6 = {"input": input6, "s": s6, "dim": dim6, "norm": norm6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.fft.irfft2"] = irfft2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.irfft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.irfft2'.")


check_valid('torch.fft.irfft2', generated_inputs['torch.fft.irfft2'], lib="torch", suffix=0)
