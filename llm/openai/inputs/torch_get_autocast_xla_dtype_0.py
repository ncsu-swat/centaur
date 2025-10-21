
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def get_autocast_xla_dtype_inputs():
    list_of_inputs = []
    
    input1 = np.float32
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input1}))
    
    input2 = np.float16
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input2}))

    input3 = np.float64
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input3}))

    input5 = np.complex64
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input5}))
    
    input6 = np.complex128
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input6}))

    input7 = np.int32
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input7}))

    input8 = np.int64
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input8}))
    
    input9 = np.uint8
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input9}))

    input10 = np.bool_
    list_of_inputs.append(copy.deepcopy({"torch.get_autocast_xla_dtype": input10}))

    return list_of_inputs

generated_inputs["torch.get_autocast_xla_dtype"] = get_autocast_xla_dtype_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.get_autocast_xla_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_autocast_xla_dtype'.")


check_valid('torch.get_autocast_xla_dtype', generated_inputs['torch.get_autocast_xla_dtype'], lib="torch", suffix=0)
