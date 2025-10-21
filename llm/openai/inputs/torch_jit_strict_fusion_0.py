
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def strict_fusion_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input1
    })
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input2
    })
    
    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input3
    })
    
    input4 = np.array([0.0], dtype=np.float16)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input4
    })
    
    input5 = np.array([], dtype=np.float32)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input5
    })
    
    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input6
    })
    
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input7
    })

    input8 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input8
    })
    
    input9 = np.array([1.0], dtype=np.float64)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input9
    })
    
    input10 = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    list_of_inputs.append({
        "torch.jit.strict_fusion": input10
    })
    

    return list_of_inputs

generated_inputs["torch.jit.strict_fusion"] = strict_fusion_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.strict_fusion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.strict_fusion'.")


check_valid('torch.jit.strict_fusion', generated_inputs['torch.jit.strict_fusion'], lib="torch", suffix=0)
