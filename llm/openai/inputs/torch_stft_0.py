
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np
from scipy.signal import windows

def torch_stft_inputs():
    list_of_inputs = []

    input1 = np.random.rand(100).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "n_fft": 256,
        "hop_length": 64,
        "win_length": 128,
        "window": np.ones(128).astype(np.float32),
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": True,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 50).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 256,
        "window": windows.hann(256).astype(np.float32),
        "center": False,
        "pad_mode": "constant",
        "normalized": True,
        "onesided": False,
        "return_complex": False,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.stft"] = torch_stft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.stft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stft'.")


check_valid('torch.stft', generated_inputs['torch.stft'], lib="torch", suffix=0)
