
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unfold_inputs():
    list_of_inputs = []

    input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
    kernel_size1 = (3, 3)
    dilation1 = 1
    padding1 = 1
    stride1 = 1
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "dilation": dilation1,
        "padding": padding1,
        "stride": stride1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3, 64, 64).astype(np.float32)
    kernel_size2 = (5, 5)
    dilation2 = 2
    padding2 = 0
    stride2 = 2
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "dilation": dilation2,
        "padding": padding2,
        "stride": stride2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(4, 3, 128, 128).astype(np.float32)
    kernel_size3 = (1, 1)
    dilation3 = 1
    padding3 = 2
    stride3 = 1
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "dilation": dilation3,
        "padding": padding3,
        "stride": stride3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 1, 32, 32).astype(np.float32)
    kernel_size4 = (7, 7)
    dilation4 = 3
    padding4 = 3
    stride4 = 3
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "dilation": dilation4,
        "padding": padding4,
        "stride": stride4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 3, 256, 256).astype(np.float32)
    kernel_size5 = (2, 2)
    dilation5 = 1
    padding5 = 1
    stride5 = 2
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "dilation": dilation5,
        "padding": padding5,
        "stride": stride5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1, 3, 100, 100).astype(np.float32)
    kernel_size6 = (3, 3)
    dilation6 = 2
    padding6 = 0
    stride6 = 1
    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "dilation": dilation6,
        "padding": padding6,
        "stride": stride6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(5, 3, 32, 32).astype(np.float32)
    kernel_size7 = (1, 1)
    dilation7 = 1
    padding7 = 0
    stride7 = 1
    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "dilation": dilation7,
        "padding": padding7,
        "stride": stride7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 3, 128, 128).astype(np.float32)
    kernel_size8 = (5, 5)
    dilation8 = 1
    padding8 = 2
    stride8 = 2
    input_dict8 = {
        "input": input8,
        "kernel_size": kernel_size8,
        "dilation": dilation8,
        "padding": padding8,
        "stride": stride8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(1, 3, 64, 64).astype(np.float32)
    kernel_size9 = (3, 3)
    dilation9 = 3
    padding9 = 1
    stride9 = 1
    input_dict9 = {
        "input": input9,
        "kernel_size": kernel_size9,
        "dilation": dilation9,
        "padding": padding9,
        "stride": stride9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(4, 3, 224, 224).astype(np.float32)
    kernel_size10 = (7, 7)
    dilation10 = 1
    padding10 = 3
    stride10 = 2
    input_dict10 = {
        "input": input10,
        "kernel_size": kernel_size10,
        "dilation": dilation10,
        "padding": padding10,
        "stride": stride10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.unfold"] = unfold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.unfold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.unfold'.")


check_valid('torch.nn.functional.unfold', generated_inputs['torch.nn.functional.unfold'], lib="torch", suffix=0)
