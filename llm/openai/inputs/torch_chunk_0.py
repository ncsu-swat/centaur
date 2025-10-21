
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def chunk_inputs():
    list_of_inputs = []

    input1 = torch.arange(10).numpy()
    chunks1 = 2
    dim1 = 0
    input_dict1 = {"input": input1, "chunks": chunks1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(12, 5).numpy()
    chunks2 = 3
    dim2 = 1
    input_dict2 = {"input": input2, "chunks": chunks2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).numpy()
    chunks3 = 4
    dim3 = 0
    input_dict3 = {"input": input3, "chunks": chunks3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4).numpy()
    chunks4 = 2
    dim4 = 2
    input_dict4 = {"input": input4, "chunks": chunks4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.arange(15).numpy()
    chunks5 = 5
    dim5 = 0
    input_dict5 = {"input": input5, "chunks": chunks5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(8, 8).numpy()
    chunks6 = 4
    dim6 = 1
    input_dict6 = {"input": input6, "chunks": chunks6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.arange(7).numpy()
    chunks7 = 3
    dim7 = 0
    input_dict7 = {"input": input7, "chunks": chunks7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(4, 6, 2).numpy()
    chunks8 = 2
    dim8 = 1
    input_dict8 = {"input": input8, "chunks": chunks8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    chunks9 = 2
    dim9 = 0
    input_dict9 = {"input": input9, "chunks": chunks9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(10, 10, 10).numpy()
    chunks10 = 5
    dim10 = 2
    input_dict10 = {"input": input10, "chunks": chunks10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.chunk"] = chunk_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.chunk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chunk'.")


check_valid('torch.chunk', generated_inputs['torch.chunk'], lib="torch", suffix=0)
