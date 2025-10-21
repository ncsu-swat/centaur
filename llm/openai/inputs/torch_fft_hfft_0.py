
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hfft_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    n1 = 5
    dim1 = 0
    norm1 = "backward"
    out1 = np.zeros(5, dtype=np.float32)
    input_dict1 = {"input": input1, "n": n1, "dim": dim1, "norm": norm1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n2 = 3
    dim2 = 0
    norm2 = "forward"
    out2 = np.zeros(3, dtype=np.float32)
    input_dict2 = {"input": input2, "n": n2, "dim": dim2, "norm": norm2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, -2.0, 1.0], dtype=np.float32)
    n3 = None
    dim3 = 0
    norm3 = "ortho"
    out3 = np.zeros(3, dtype=np.float32)
    input_dict3 = {"input": input3, "n": n3, "dim": dim3, "norm": norm3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n4 = 4
    dim4 = 1
    norm4 = "backward"
    out4 = np.zeros((2, 2), dtype=np.float32)
    input_dict4 = {"input": input4, "n": n4, "dim": dim4, "norm": norm4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    n5 = 8
    dim5 = 0
    norm5 = "forward"
    out5 = np.zeros(8, dtype=np.float32)
    input_dict5 = {"input": input5, "n": n5, "dim": dim5, "norm": norm5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    n6 = None
    dim6 = 0
    norm6 = "ortho"
    out6 = np.zeros(4, dtype=np.float32)
    input_dict6 = {"input": input6, "n": n6, "dim": dim6, "norm": norm6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    n7 = 4
    dim7 = 0
    norm7 = "backward"
    out7 = np.zeros(4, dtype=np.float32)
    input_dict7 = {"input": input7, "n": n7, "dim": dim7, "norm": norm7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.fft.hfft"] = hfft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.hfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.hfft'.")


check_valid('torch.fft.hfft', generated_inputs['torch.fft.hfft'], lib="torch", suffix=0)
