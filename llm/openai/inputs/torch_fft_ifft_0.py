
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ifft_inputs():
    list_of_inputs = []

    input1 = torch.tensor([6. + 0.j, -2. + 2.j, -2. + 0.j, -2. - 2.j]).numpy()
    n1 = 4
    dim1 = 0
    norm1 = "backward"
    out1 = torch.zeros(4, dtype=torch.complex64).numpy()

    input_dict1 = {
        "input": input1,
        "n": n1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2, dtype=torch.complex128).numpy()
    n2 = None
    dim2 = 1
    norm2 = "ortho"
    out2 = torch.zeros((2, 2), dtype=torch.complex128).numpy()

    input_dict2 = {
        "input": input2,
        "n": n2,
        "dim": dim2,
        "norm": norm2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j]).numpy()
    n3 = 8
    dim3 = 0
    norm3 = "forward"
    out3 = torch.zeros(8, dtype=torch.complex64).numpy()

    input_dict3 = {
        "input": input3,
        "n": n3,
        "dim": dim3,
        "norm": norm3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, dtype=torch.complex64).numpy()
    n4 = None
    dim4 = 0
    norm4 = "backward"
    out4 = torch.zeros(4, dtype=torch.complex64).numpy()

    input_dict4 = {
        "input": input4,
        "n": n4,
        "dim": dim4,
        "norm": norm4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j]).numpy()
    n5 = 4
    dim5 = 0
    norm5 = "ortho"
    out5 = torch.zeros(4, dtype=torch.complex64).numpy()

    input_dict5 = {
        "input": input5,
        "n": n5,
        "dim": dim5,
        "norm": norm5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 3, dtype=torch.complex128).numpy()
    n6 = None
    dim6 = 1
    norm6 = "backward"
    out6 = torch.zeros((2, 3), dtype=torch.complex128).numpy()

    input_dict6 = {
        "input": input6,
        "n": n6,
        "dim": dim6,
        "norm": norm6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([5.0 + 0.0j, 6.0 + 0.0j]).numpy()
    n7 = 8
    dim7 = 0
    norm7 = "forward"
    out7 = torch.zeros(8, dtype=torch.complex64).numpy()

    input_dict7 = {
        "input": input7,
        "n": n7,
        "dim": dim7,
        "norm": norm7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(5, dtype=torch.complex64).numpy()
    n8 = None
    dim8 = 0
    norm8 = "ortho"
    out8 = torch.zeros(5, dtype=torch.complex64).numpy()

    input_dict8 = {
        "input": input8,
        "n": n8,
        "dim": dim8,
        "norm": norm8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1.0 + 1.0j, -2.0 + 2.0j]).numpy()
    n9 = 4
    dim9 = 0
    norm9 = "backward"
    out9 = torch.zeros(4, dtype=torch.complex64).numpy()

    input_dict9 = {
        "input": input9,
        "n": n9,
        "dim": dim9,
        "norm": norm9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(3, 4, dtype=torch.complex128).numpy()
    n10 = None
    dim10 = 1
    norm10 = "forward"
    out10 = torch.zeros((3, 4), dtype=torch.complex128).numpy()

    input_dict10 = {
        "input": input10,
        "n": n10,
        "dim": dim10,
        "norm": norm10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.fft.ifft"] = ifft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.ifft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifft'.")


check_valid('torch.fft.ifft', generated_inputs['torch.fft.ifft'], lib="torch", suffix=0)
