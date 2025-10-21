
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fake_quantize_per_tensor_affine_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale1 = 0.1
    zero_point1 = 0
    quant_min1 = -10
    quant_max1 = 10
    
    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "quant_min": quant_min1,
        "quant_max": quant_max1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale2 = 0.2
    zero_point2 = 1
    quant_min2 = -5
    quant_max2 = 5
    
    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "quant_min": quant_min2,
        "quant_max": quant_max2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    scale3 = 0.5
    zero_point3 = -1
    quant_min3 = -20
    quant_max3 = 20
    
    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "quant_min": quant_min3,
        "quant_max": quant_max3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0], dtype=np.float32)
    scale4 = 1.0
    zero_point4 = 0
    quant_min4 = 0
    quant_max4 = 100
    
    input_dict4 = {
        "input": input4,
        "scale": scale4,
        "zero_point": zero_point4,
        "quant_min": quant_min4,
        "quant_max": quant_max4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    scale5 = 0.3
    zero_point5 = 2
    quant_min5 = -10
    quant_max5 = 10
    
    input_dict5 = {
        "input": input5,
        "scale": scale5,
        "zero_point": zero_point5,
        "quant_min": quant_min5,
        "quant_max": quant_max5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    scale6 = 0.4
    zero_point6 = -2
    quant_min6 = -50
    quant_max6 = 50
    
    input_dict6 = {
        "input": input6,
        "scale": scale6,
        "zero_point": zero_point6,
        "quant_min": quant_min6,
        "quant_max": quant_max6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    scale7 = 0.7
    zero_point7 = 1
    quant_min7 = -20
    quant_max7 = 20
    
    input_dict7 = {
        "input": input7,
        "scale": scale7,
        "zero_point": zero_point7,
        "quant_min": quant_min7,
        "quant_max": quant_max7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float32)
    scale8 = 1.0
    zero_point8 = 0
    quant_min8 = -10
    quant_max8 = 10
    
    input_dict8 = {
        "input": input8,
        "scale": scale8,
        "zero_point": zero_point8,
        "quant_min": quant_min8,
        "quant_max": quant_max8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    scale9 = 0.6
    zero_point9 = -1
    quant_min9 = -15
    quant_max9 = 15
    
    input_dict9 = {
        "input": input9,
        "scale": scale9,
        "zero_point": zero_point9,
        "quant_min": quant_min9,
        "quant_max": quant_max9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale10 = 0.1
    zero_point10 = 0
    quant_min10 = -2
    quant_max10 = 2
    
    input_dict10 = {
        "input": input10,
        "scale": scale10,
        "zero_point": zero_point10,
        "quant_min": quant_min10,
        "quant_max": quant_max10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.fake_quantize_per_tensor_affine"] = fake_quantize_per_tensor_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fake_quantize_per_tensor_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_tensor_affine'.")


check_valid('torch.fake_quantize_per_tensor_affine', generated_inputs['torch.fake_quantize_per_tensor_affine'], lib="torch", suffix=0)
