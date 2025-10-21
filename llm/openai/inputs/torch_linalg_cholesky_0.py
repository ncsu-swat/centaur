
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_inputs():
    list_of_inputs = []
    
    A1 = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
    upper1 = False
    out1 = np.empty((3, 3), dtype=np.float64)
    input_dict1 = {"A": A1, "upper": upper1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    A2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    upper2 = True
    out2 = np.empty((3, 3), dtype=np.float32)
    input_dict2 = {"A": A2, "upper": upper2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    A3 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.complex128)
    upper3 = False
    out3 = np.empty((3, 3), dtype=np.complex128)
    input_dict3 = {"A": A3, "upper": upper3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    A4 = np.array([[5, 2, 1], [2, 8, 3], [1, 3, 6]], dtype=np.float64)
    upper4 = True
    out4 = np.empty((3, 3), dtype=np.float64)
    input_dict4 = {"A": A4, "upper": upper4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    A5 = np.random.rand(2, 2)
    A5 = A5 @ A5.T + np.eye(2)
    upper5 = False
    out5 = np.empty((2, 2))
    input_dict5 = {"A": A5, "upper": upper5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    A6 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    upper6 = False
    out6 = np.empty((2, 2), dtype=np.float64)
    input_dict6 = {"A": A6, "upper": upper6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    A7 = np.array([[7, 3, -1], [3, 10, 2], [-1, 2, 5]], dtype=np.float32)
    upper7 = True
    out7 = np.empty((3, 3), dtype=np.float32)
    input_dict7 = {"A": A7, "upper": upper7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    A8 = np.random.rand(4, 4)
    A8 = A8 @ A8.T + np.eye(4)
    upper8 = False
    out8 = np.empty((4, 4))
    input_dict8 = {"A": A8, "upper": upper8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    A9 = np.array([[1, 2, 3], [2, 5, 7], [3, 7, 11]], dtype=np.float64)
    upper9 = True
    out9 = np.empty((3, 3), dtype=np.float64)
    input_dict9 = {"A": A9, "upper": upper9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    A10 = np.array([[10, -5, 2], [-5, 10, -1], [2, -1, 10]], dtype=np.complex128)
    upper10 = False
    out10 = np.empty((3, 3), dtype=np.complex128)
    input_dict10 = {"A": A10, "upper": upper10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.linalg.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.cholesky'.")


check_valid('torch.linalg.cholesky', generated_inputs['torch.linalg.cholesky'], lib="torch", suffix=0)
