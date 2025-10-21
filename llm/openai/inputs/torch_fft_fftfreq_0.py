
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fftfreq_inputs():
    list_of_inputs = []
    
    input1 = np.int32(5)
    d1 = np.float64(1.0)
    out1 = np.zeros(5, dtype=np.float64)
    dtype1 = torch.float64
    requires_grad1 = False
    
    input_dict1 = {
        "n": input1,
        "d": d1,
        "out": out1,
        "dtype": dtype1,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.int32(10)
    d2 = np.float64(0.5)
    out2 = np.zeros(10, dtype=np.float32)
    dtype2 = torch.float32
    requires_grad2 = True
    
    input_dict2 = {
        "n": input2,
        "d": d2,
        "out": out2,
        "dtype": dtype2,
        "requires_grad": requires_grad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.int32(7)
    d3 = np.float64(2.0)
    out3 = np.zeros(7, dtype=np.float64)
    dtype3 = torch.float64
    requires_grad3 = False
    
    input_dict3 = {
        "n": input3,
        "d": d3,
        "out": out3,
        "dtype": dtype3,
        "requires_grad": requires_grad3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.int32(12)
    d4 = np.float64(1.5)
    out4 = np.zeros(12, dtype=np.float32)
    dtype4 = torch.float32
    requires_grad4 = True
    
    input_dict4 = {
        "n": input4,
        "d": d4,
        "out": out4,
        "dtype": dtype4,
        "requires_grad": requires_grad4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.int32(3)
    d5 = np.float64(0.25)
    out5 = np.zeros(3, dtype=np.float64)
    dtype5 = torch.float64
    requires_grad5 = False
    
    input_dict5 = {
        "n": input5,
        "d": d5,
        "out": out5,
        "dtype": dtype5,
        "requires_grad": requires_grad5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    
    return list_of_inputs

generated_inputs["torch.fft.fftfreq"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.fftfreq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftfreq'.")


check_valid('torch.fft.fftfreq', generated_inputs['torch.fft.fftfreq'], lib="torch", suffix=0)
