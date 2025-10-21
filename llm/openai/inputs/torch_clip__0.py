
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min1 = np.float32(0.0)
    max1 = np.float32(2.5)
    input_dict1 = {"input": input1, "min": min1, "max": max1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    min2 = np.float32(-0.5)
    max2 = np.float32(0.5)
    input_dict2 = {"input": input2, "min": min2, "max": max2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    min3 = np.float32(1.5)
    max3 = np.float32(3.5)
    input_dict3 = {"input": input3, "min": min3, "max": max3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    min4 = np.float64(4.0)
    max4 = np.float64(6.5)
    input_dict4 = {"input": input4, "min": min4, "max": max4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([], dtype=np.float32)
    min5 = np.float32(0.0)
    max5 = np.float32(1.0)
    input_dict5 = {"input": input5, "min": min5, "max": max5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    min6 = np.float32(-3.0)
    max6 = np.float32(-1.5)
    input_dict6 = {"input": input6, "min": min6, "max": max6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min7 = np.float32(2.0)
    max7 = np.float32(4.0)
    input_dict7 = {"input": input7, "min": min7, "max": max7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    min8 = np.float64(5.0)
    max8 = np.float64(7.0)
    input_dict8 = {"input": input8, "min": min8, "max": max8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.1, 0.9, 0.5, 0.2], dtype=np.float32)
    min9 = np.float32(0.2)
    max9 = np.float32(0.8)
    input_dict9 = {"input": input9, "min": min9, "max": max9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([10.0], dtype=np.float32)
    min10 = np.float32(5.0)
    max10 = np.float32(15.0)
    input_dict10 = {"input": input10, "min": min10, "max": max10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.clip_"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clip_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip_'.")


check_valid('torch.clip_', generated_inputs['torch.clip_'], lib="torch", suffix=0)
