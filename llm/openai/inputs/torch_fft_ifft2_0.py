
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ifft2_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(4, 4).astype(np.complex64)
    s1 = (4, 4)
    dim1 = (-2, -1)
    norm1 = "backward"
    out1 = np.zeros_like(input1)
    
    input_dict1 = {
        "input": input1,
        "s": s1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 2, 2, 2).astype(np.complex64)
    s2 = (2, 2)
    dim2 = (-2, -1)
    norm2 = "forward"
    out2 = np.zeros_like(input2)
    
    input_dict2 = {
        "input": input2,
        "s": s2,
        "dim": dim2,
        "norm": norm2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(8, 8).astype(np.complex128)
    s3 = (8, 8)
    dim3 = (-2, -1)
    norm3 = "ortho"
    out3 = np.zeros_like(input3)
    
    input_dict3 = {
        "input": input3,
        "s": s3,
        "dim": dim3,
        "norm": norm3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(16, 16).astype(np.complex64)
    s4 = (16, 16)
    dim4 = (-2, -1)
    norm4 = "backward"
    out4 = np.zeros_like(input4)
    
    input_dict4 = {
        "input": input4,
        "s": s4,
        "dim": dim4,
        "norm": norm4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(4, 4).astype(np.complex64)
    s5 = (2, 2)
    dim5 = (-2, -1)
    norm5 = "backward"
    out5 = np.zeros_like(input5)
    
    input_dict5 = {
        "input": input5,
        "s": s5,
        "dim": dim5,
        "norm": norm5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(8, 8).astype(np.complex64)
    s6 = (8, 8)
    dim6 = (0, 1)
    norm6 = "backward"
    out6 = np.zeros_like(input6)

    input_dict6 = {
        "input": input6,
        "s": s6,
        "dim": dim6,
        "norm": norm6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4).astype(np.complex64)
    s7 = (4, 4)
    dim7 = (-2, -1)
    norm7 = "backward"
    out7 = np.zeros_like(input7)

    input_dict7 = {
        "input": input7,
        "s": s7,
        "dim": dim7,
        "norm": norm7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.random.rand(2, 2, 2, 2).astype(np.complex128)
    s8 = (2, 2)
    dim8 = (-2, -1)
    norm8 = "ortho"
    out8 = np.zeros_like(input8)

    input_dict8 = {
        "input": input8,
        "s": s8,
        "dim": dim8,
        "norm": norm8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(16, 16).astype(np.complex64)
    s9 = (16, 16)
    dim9 = (-2, -1)
    norm9 = "forward"
    out9 = np.zeros_like(input9)

    input_dict9 = {
        "input": input9,
        "s": s9,
        "dim": dim9,
        "norm": norm9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(8, 8).astype(np.complex64)
    s10 = (8, 8)
    dim10 = (-2, -1)
    norm10 = "backward"
    out10 = np.zeros_like(input10)
    
    input_dict10 = {
        "input": input10,
        "s": s10,
        "dim": dim10,
        "norm": norm10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.fft.ifft2"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.ifft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifft2'.")


check_valid('torch.fft.ifft2', generated_inputs['torch.fft.ifft2'], lib="torch", suffix=0)
