
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rfft2_inputs():
    list_of_inputs = []

    input1 = np.random.rand(10, 10).astype(np.float32)
    s1 = (10, 10)
    dim1 = (-2, -1)
    norm1 = "backward"
    out1 = np.zeros((10, 6), dtype=np.complex64)

    input_dict1 = {
        "input": input1,
        "s": s1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(8, 8).astype(np.float64)
    s2 = (8, 8)
    dim2 = (-2, -1)
    norm2 = "forward"
    out2 = np.zeros((8, 4), dtype=np.complex64)

    input_dict2 = {
        "input": input2,
        "s": s2,
        "dim": dim2,
        "norm": norm2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(4, 4).astype(np.float32)
    s3 = (4, 4)
    dim3 = (-2, -1)
    norm3 = "ortho"
    out3 = np.zeros((4, 3), dtype=np.complex64)

    input_dict3 = {
        "input": input3,
        "s": s3,
        "dim": dim3,
        "norm": norm3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(16, 16).astype(np.float32)
    s4 = (16, 16)
    dim4 = (0, 1)
    norm4 = "backward"
    out4 = np.zeros((16, 8), dtype=np.complex64)

    input_dict4 = {
        "input": input4,
        "s": s4,
        "dim": dim4,
        "norm": norm4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(32, 32).astype(np.float64)
    s5 = (32, 32)
    dim5 = (-2, -1)
    norm5 = "forward"
    out5 = np.zeros((32, 16), dtype=np.complex64)

    input_dict5 = {
        "input": input5,
        "s": s5,
        "dim": dim5,
        "norm": norm5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(64, 64).astype(np.float32)
    s6 = (64, 64)
    dim6 = (-2, -1)
    norm6 = "ortho"
    out6 = np.zeros((64, 32), dtype=np.complex64)

    input_dict6 = {
        "input": input6,
        "s": s6,
        "dim": dim6,
        "norm": norm6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(5, 5).astype(np.float32)
    s7 = (5, 5)
    dim7 = (-2, -1)
    norm7 = "backward"
    out7 = np.zeros((5, 3), dtype=np.complex64)

    input_dict7 = {
        "input": input7,
        "s": s7,
        "dim": dim7,
        "norm": norm7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.random.rand(20, 20).astype(np.float64)
    s8 = (20, 20)
    dim8 = (-2, -1)
    norm8 = "forward"
    out8 = np.zeros((20, 10), dtype=np.complex64)

    input_dict8 = {
        "input": input8,
        "s": s8,
        "dim": dim8,
        "norm": norm8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(12, 12).astype(np.float32)
    s9 = (12, 12)
    dim9 = (-2, -1)
    norm9 = "ortho"
    out9 = np.zeros((12, 6), dtype=np.complex64)

    input_dict9 = {
        "input": input9,
        "s": s9,
        "dim": dim9,
        "norm": norm9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(7, 7).astype(np.float64)
    s10 = (7, 7)
    dim10 = (-2, -1)
    norm10 = "backward"
    out10 = np.zeros((7, 4), dtype=np.complex64)

    input_dict10 = {
        "input": input10,
        "s": s10,
        "dim": dim10,
        "norm": norm10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.fft.rfft2"] = rfft2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.rfft2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfft2'.")


check_valid('torch.fft.rfft2', generated_inputs['torch.fft.rfft2'], lib="torch", suffix=0)
