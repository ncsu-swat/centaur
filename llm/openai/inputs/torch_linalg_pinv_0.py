
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pinv_inputs():
    list_of_inputs = []

    input_1 = torch.randn(3, 5).numpy()
    atol_1 = 1e-8
    rtol_1 = 1e-5
    hermitian_1 = False
    out_1 = np.zeros((3, 5))
    input_dict_1 = {"A": input_1, "atol": atol_1, "rtol": rtol_1, "hermitian": hermitian_1, "out": out_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 6, 3).numpy()
    atol_2 = 0.0
    rtol_2 = 1e-3
    hermitian_2 = False
    out_2 = np.zeros((2, 6, 3))
    input_dict_2 = {"A": input_2, "atol": atol_2, "rtol": rtol_2, "hermitian": hermitian_2, "out": out_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(3, 3, dtype=torch.complex64).numpy()
    atol_3 = 1e-6
    rtol_3 = 0.0
    hermitian_3 = True
    out_3 = np.zeros((3, 3), dtype=np.complex64)
    input_dict_3 = {"A": input_3, "atol": atol_3, "rtol": rtol_3, "hermitian": hermitian_3, "out": out_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(5, 2).numpy()
    atol_4 = 1e-7
    rtol_4 = 1e-4
    hermitian_4 = False
    out_4 = np.zeros((5, 2))
    input_dict_4 = {"A": input_4, "atol": atol_4, "rtol": rtol_4, "hermitian": hermitian_4, "out": out_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.randn(4, 4).numpy()
    atol_5 = 0.0
    rtol_5 = 1e-2
    hermitian_5 = False
    out_5 = np.zeros((4, 4))
    input_dict_5 = {"A": input_5, "atol": atol_5, "rtol": rtol_5, "hermitian": hermitian_5, "out": out_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = torch.randn(2, 2, dtype=torch.complex128).numpy()
    atol_6 = 1e-9
    rtol_6 = 1e-6
    hermitian_6 = True
    out_6 = np.zeros((2, 2), dtype=np.complex128)
    input_dict_6 = {"A": input_6, "atol": atol_6, "rtol": rtol_6, "hermitian": hermitian_6, "out": out_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = torch.randn(1, 7).numpy()
    atol_7 = 1e-10
    rtol_7 = 1e-7
    hermitian_7 = False
    out_7 = np.zeros((1, 7))
    input_dict_7 = {"A": input_7, "atol": atol_7, "rtol": rtol_7, "hermitian": hermitian_7, "out": out_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    input_8 = torch.randn(6, 1).numpy()
    atol_8 = 0.0
    rtol_8 = 1e-1
    hermitian_8 = False
    out_8 = np.zeros((6, 1))
    input_dict_8 = {"A": input_8, "atol": atol_8, "rtol": rtol_8, "hermitian": hermitian_8, "out": out_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_9 = torch.randn(2, 3, 4).numpy()
    atol_9 = 1e-8
    rtol_9 = 1e-5
    hermitian_9 = False
    out_9 = np.zeros((2, 3, 4))
    input_dict_9 = {"A": input_9, "atol": atol_9, "rtol": rtol_9, "hermitian": hermitian_9, "out": out_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = torch.randn(3, 2).numpy()
    atol_10 = 1e-6
    rtol_10 = 0.0
    hermitian_10 = False
    out_10 = np.zeros((3, 2))
    input_dict_10 = {"A": input_10, "atol": atol_10, "rtol": rtol_10, "hermitian": hermitian_10, "out": out_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.linalg.pinv"] = pinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.pinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.pinv'.")


check_valid('torch.linalg.pinv', generated_inputs['torch.linalg.pinv'], lib="torch", suffix=0)
