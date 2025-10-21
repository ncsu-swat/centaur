
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def prepare_multiprocessing_environment_inputs():
    list_of_inputs = []
    
    input1 = "default"
    input_dict1 = {"file_system": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = "posix"
    input_dict2 = {"file_system": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = "nt"
    input_dict3 = {"file_system": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = "relative_path"
    input_dict4 = {"file_system": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = "absolute_path"
    input_dict5 = {"file_system": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = "mixed_path"
    input_dict6 = {"file_system": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = "long_string_path" + str(np.random.randint(0, 100))
    input_dict7 = {"file_system": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = "short_string"
    input_dict8 = {"file_system": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = "path_with_spaces"
    input_dict9 = {"file_system": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = "special_chars!@#$"
    input_dict10 = {"file_system": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    input11 = "path/with/slashes"
    input_dict11 = {"file_system": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs["torch.prepare_multiprocessing_environment"] = prepare_multiprocessing_environment_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.prepare_multiprocessing_environment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prepare_multiprocessing_environment'.")


check_valid('torch.prepare_multiprocessing_environment', generated_inputs['torch.prepare_multiprocessing_environment'], lib="torch", suffix=0)
